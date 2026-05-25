"""Pipeline generator driver.

Single source of truth for "produce one product image". The driver
orchestrates four pluggable seams:

  - PromptRegistry: which YAML template to render (anchor or edit)
  - SourceProvider: where the source image comes from (edit mode only)
  - Backend:        the model that turns request → bytes
  - HistoryStore:   snapshot before any on-disk overwrite

Flow per variant:

  1. Load incidental detail tag(s) from pricing_naming.json.
  2. build_prompt(spec) — substitutes form, color, material, detail.
  3. If edit-mode: source_provider.source_for(spec) → bytes.
  4. backend.generate(GenerateRequest(prompt, seed, source_image)) → bytes
  5. QC: CLIP text-image cosine ≥ threshold AND Gemini vision check
  6. On failure: retry up to MAX_RETRIES. After all fail: write the
     last image with qc_status='manual_review'.
  7. Before writing: history.snapshot_before_write(target, current_tag)
  8. Write PNG → public/<imagePath>.
  9. Append manifest record (with mode, source_image_sha256, etc.).

Usage:

  # text-to-image with Gemini (Phase A default — unchanged behaviour)
  python3 -m pipeline.generate --all

  # explicit
  python3 -m pipeline.generate --all --backend gemini --mode text2image

  # edit-refine current on-disk PNGs with another backend
  python3 -m pipeline.generate --all --backend qwen-image-edit \\
      --mode edit --source existing-file

Other flags:
  --only <urlHash>   single variant
  --limit N          cap successful generations
  --dry              print prompt, no API call
  --resume           skip variants whose latest manifest record is 'ok'
  --style <list>     filter variants by style (comma-separated)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from .backends import (
    BackendError,
    BackendUnsupportedError,
    GenerateRequest,
    ImageBackend,
    make_backend,
)
from .history import snapshot_before_write
from .manifest import ManifestRecord, ManifestWriter, iso_now, sha256_bytes
from .prompts import PromptConfig, load_anchor, load_edit
from .qc.clip_text import CLIP_TEXT_THRESHOLD, clip_text_image_cosine
from .qc.vision import detail_visible


REPO = Path(__file__).resolve().parents[1]
PRICING_PATH = REPO / "VisMem-Diag" / "env" / "scripts" / "pricing_naming.json"
IMAGING_SPECS = REPO / "VisMem-Diag" / "env" / "scripts" / "imaging_specs.json"
PUBLIC_ROOT = REPO / "VisMem-Diag" / "env" / "frontend" / "public"

MAX_RETRIES = 3
MIN_BYTES = 1024


def stable_seed_for(url_hash: str) -> int:
    """Deterministic 32-bit seed derived from urlHash."""
    h = hashlib.sha256(url_hash.encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7fffffff


def load_variant_specs() -> list[dict[str, Any]]:
    imaging = json.loads(IMAGING_SPECS.read_text())
    pricing = json.loads(PRICING_PATH.read_text())["variants"]
    out: list[dict[str, Any]] = []
    for urlhash, ispec in imaging.items():
        pv = pricing.get(urlhash) or {}
        details = pv.get("incidentalDetails") or []
        if not details:
            raise RuntimeError(
                f"variant {urlhash} has no incidentalDetails. "
                f"Run `python3 tools/tag_incidental_details.py` first."
            )
        out.append({
            "urlHash": urlhash,
            "categorySlug": ispec["categorySlug"],
            "noun": ispec["noun"],
            "style": ispec["style"],
            "color": ispec["color"],
            "imagePath": ispec["imagePath"],
            "material": pv.get("materialDescriptor") or _default_material(ispec["style"]),
            "incidentalDetails": details,
        })
    return out


def _default_material(style: str) -> str:
    return {
        "modern":     "woven fabric or wood",
        "minimalist": "matte panel or smooth fabric",
        "vintage":    "tufted fabric or aged wood",
        "industrial": "raw steel or rough fabric",
    }.get(style, "blended fabric")


def resolve_target(image_path: str, output_root: Path | None = None) -> Path:
    """Map a variant's `imagePath` (e.g. '/images/vase/var_a_t1.png')
    to a filesystem target. By default it lands at
    `<PUBLIC_ROOT>/images/vase/var_a_t1.png` (the live image tree
    served by the frontend). When `output_root` is supplied, the
    leading `/images/` prefix is replaced with `<output_root>/` —
    used to redirect a model's output into a parallel directory
    (e.g. `images_Gemini/` for the Gemini-only reference set)
    without disturbing the canonical `images/` tree.
    """
    if output_root is not None:
        root = Path(output_root)
        if not root.is_absolute():
            root = (REPO / root).resolve()
        rel = image_path.lstrip("/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        return root / rel
    return PUBLIC_ROOT / image_path.lstrip("/")


def _qc(image_bytes: bytes, prompt: str, detail_phrase: str,
        api_key: str | None) -> dict[str, Any]:
    cos = clip_text_image_cosine(image_bytes, prompt)
    cos_ok = cos >= CLIP_TEXT_THRESHOLD
    vision = detail_visible(image_bytes, detail_phrase, api_key=api_key)
    return {
        "clip_text_cosine": cos,
        "clip_text_passed": cos_ok,
        "vision_answer": vision.answer,
        "vision_passed": vision.passed,
        "overall_passed": cos_ok and vision.passed,
    }


def generate_one(
    spec: dict[str, Any],
    *,
    backend: ImageBackend,
    config: PromptConfig,
    writer: ManifestWriter,
    api_key: str | None,
    dry: bool,
    mode: str = "text2image",
    source_provider: Any = None,            # SourceProvider when mode='edit'
    qc_detail_phrase_config: PromptConfig | None = None,
    latest_tag: str | None = None,           # prompt_hash of file to be overwritten
    output_root: Path | None = None,         # redirect target out of PUBLIC_ROOT
) -> str:
    """Generate + QC + manifest one variant.

    Returns: qc_status ('ok' | 'manual_review' | 'backend_failed' | 'dry').

    qc_detail_phrase_config: the PromptConfig used to compute the
    detail-phrase for QC; in edit mode this should be the anchor config
    (since the rich description is what vision QC checks against), even
    though the model receives the shorter edit-instruction prompt.
    Defaults to `config` for backwards compatibility with t2i runs.
    """
    prompt = config.build_prompt(spec)
    qc_cfg = qc_detail_phrase_config or config
    detail_phrase = qc_cfg.detail_phrase_for(spec["incidentalDetails"])
    target = resolve_target(spec["imagePath"], output_root=output_root)

    source_image: bytes | None = None
    source_sha = ""
    source_provider_id = ""
    if mode == "edit":
        if source_provider is None:
            raise ValueError("edit mode requires a source_provider")
        source_image = source_provider.source_for(spec)
        source_sha = sha256_bytes(source_image)
        source_provider_id = source_provider.provider_id

    if dry:
        try:
            disp = target.relative_to(REPO)
        except ValueError:
            disp = target
        print(f"\n=== {spec['urlHash']} → {disp} ===")
        print(f"mode={mode} backend={backend.backend_id} "
              f"source={source_provider_id or '-'} "
              f"source_sha={source_sha[:12] if source_sha else '-'}")
        print(prompt)
        return "dry"

    base_seed = stable_seed_for(spec["urlHash"])
    last_qc: dict[str, Any] = {}
    last_bytes: bytes | None = None
    last_retry = 0
    for attempt in range(MAX_RETRIES + 1):
        seed = base_seed + attempt
        last_retry = attempt
        try:
            req = GenerateRequest(
                prompt=prompt, seed=seed, source_image=source_image,
                negative_prompt=config.negative_prompt or None,
            )
            img_bytes = backend.generate(req)
        except BackendUnsupportedError:
            # Permanent mismatch (backend doesn't do this mode);
            # let it propagate so the run halts fast.
            raise
        except BackendError as e:
            print(f"  [{spec['urlHash']}] backend fail attempt {attempt}: {e}",
                  file=sys.stderr)
            time.sleep(2.0)
            continue
        if len(img_bytes) < MIN_BYTES:
            print(f"  [{spec['urlHash']}] backend returned {len(img_bytes)}B — too small",
                  file=sys.stderr)
            continue
        last_bytes = img_bytes
        try:
            last_qc = _qc(img_bytes, prompt, detail_phrase, api_key)
        except Exception as e:
            print(f"  [{spec['urlHash']}] QC infra fail: {e}", file=sys.stderr)
            last_qc = {
                "clip_text_cosine": None,
                "clip_text_passed": False,
                "vision_answer": "",
                "vision_passed": False,
                "overall_passed": False,
                "infra_error": str(e),
            }
        if last_qc.get("overall_passed"):
            break
        print(
            f"  [{spec['urlHash']}] QC fail attempt {attempt}: "
            f"cos={last_qc.get('clip_text_cosine')} "
            f"vision={last_qc.get('vision_answer')!r} — retrying",
            file=sys.stderr,
        )

    if last_bytes is None:
        writer.append(ManifestRecord(
            urlHash=spec["urlHash"],
            prompt_hash=config.prompt_hash,
            backend_id=backend.backend_id,
            model_revision=backend.model_revision,
            seed=base_seed,
            timestamp=iso_now(),
            image_sha256="",
            incidental_details=spec["incidentalDetails"],
            retry_count=last_retry,
            qc_status="backend_failed",
            qc_clip_text_cosine=None,
            qc_vision_check={"answer": "", "passed": False},
            notes="all retries raised BackendError",
            mode=mode,
            source_image_sha256=source_sha,
            source_provider=source_provider_id,
        ))
        return "backend_failed"

    target.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot the file that's about to be overwritten so the user can
    # roll back any regen later. The tag is the prompt_hash of the
    # record that produced the current file at `target`; if there's no
    # prior record, snapshot_before_write falls back to "pre-history".
    snapshot_before_write(target, latest_tag)
    target.write_bytes(last_bytes)
    qc_status = "ok" if last_qc.get("overall_passed") else "manual_review"
    writer.append(ManifestRecord(
        urlHash=spec["urlHash"],
        prompt_hash=config.prompt_hash,
        backend_id=backend.backend_id,
        model_revision=backend.model_revision,
        seed=base_seed,
        timestamp=iso_now(),
        image_sha256=sha256_bytes(last_bytes),
        incidental_details=spec["incidentalDetails"],
        retry_count=last_retry,
        qc_status=qc_status,
        qc_clip_text_cosine=last_qc.get("clip_text_cosine"),
        qc_vision_check={
            "answer": last_qc.get("vision_answer", ""),
            "passed": last_qc.get("vision_passed", False),
            "detail_phrase": detail_phrase,
        },
        mode=mode,
        source_image_sha256=source_sha,
        source_provider=source_provider_id,
    ))
    return qc_status


def _build_source_provider(name: str, source_root: Path | None) -> Any:
    """Construct a SourceProvider by name. Kept inline to avoid an
    import cycle with pipeline.sources (which itself imports backends)."""
    if name == "existing-file":
        from .sources.file import ExistingFileSourceProvider
        return ExistingFileSourceProvider(source_root or PUBLIC_ROOT)
    if name == "vwa":
        from .sources.vwa import VWASourceProvider
        return VWASourceProvider(source_root)
    if name == "fixed-file":
        # source_root is repurposed here to carry the single file path.
        if source_root is None:
            raise ValueError(
                "fixed-file source requires --source-file <path>"
            )
        from .sources.fixed import FixedFileSourceProvider
        return FixedFileSourceProvider(source_root)
    if name == "backend":
        # Wired up properly in main() so we can pass the secondary
        # backend object directly. This branch is unreachable.
        raise NotImplementedError(
            "use --source backend with --source-backend; main() wires it"
        )
    raise ValueError(f"unknown source: {name!r}")


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="regenerate every variant from imaging_specs.json")
    ap.add_argument("--only", help="single urlHash")
    ap.add_argument("--style", help="filter to one or more styles "
                    "(comma-separated, e.g. 'minimalist' or 'minimalist,industrial')")
    ap.add_argument("--category", help="filter to one or more categorySlug values "
                    "(comma-separated, e.g. 'vases' or 'vases,plant_pots')")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap successful generations (after --resume skip)")
    ap.add_argument("--dry", action="store_true",
                    help="print prompts, don't call the API")
    ap.add_argument("--resume", action="store_true",
                    help="skip variants already present in the manifest")
    ap.add_argument("--backend", default="gemini",
                    help="which backend to use (default: gemini)")
    ap.add_argument("--mode", choices=["text2image", "edit"],
                    help="generation mode; defaults to text2image, or edit "
                    "if backend doesn't support text2image")
    ap.add_argument("--source",
                    choices=["existing-file", "vwa", "fixed-file", "backend"],
                    help="where source images for --mode=edit come from")
    ap.add_argument("--source-backend",
                    help="backend name when --source=backend "
                    "(e.g. 'gemini' to feed Qwen-edit from Gemini t2i)")
    ap.add_argument("--source-root",
                    help="filesystem root when --source=existing-file "
                    "(defaults to public/ — typically the live image tree)")
    ap.add_argument("--source-file",
                    help="single source PNG path when --source=fixed-file "
                    "(one image used as seed for every spec, e.g. for the "
                    "anchor+variant pattern)")
    ap.add_argument("--exclude",
                    help="comma-separated urlHashes to skip "
                    "(e.g. to omit the anchor variant from a category run)")
    ap.add_argument("--output-root",
                    help="redirect output PNGs into this directory "
                    "instead of public/images/ — used to keep a "
                    "parallel reference set (e.g. images_Gemini/) "
                    "without touching the canonical images/ tree")
    args = ap.parse_args()
    if not (args.all or args.only):
        print("must pass --all or --only <urlHash>", file=sys.stderr)
        return 2

    api_key = os.environ.get("GEMINI_API_KEY")

    # Construct the primary backend (used to produce output bytes).
    if args.dry:
        class _DryBackend:
            backend_id = "dry"
            model_revision = "dry"
            capabilities = frozenset({"text2image", "edit"})
            def generate(self, req):  # noqa: ARG002
                raise RuntimeError("dry-run backend should never be called")
        backend: ImageBackend = _DryBackend()  # type: ignore[assignment]
    else:
        backend = make_backend(args.backend)

    # Decide mode: explicit flag wins, else pick the backend's
    # preferred mode (text2image if supported, else edit).
    if args.mode:
        mode = args.mode
    elif "text2image" in backend.capabilities:
        mode = "text2image"
    elif "edit" in backend.capabilities:
        mode = "edit"
    else:
        print(f"backend {backend.backend_id} reports no capabilities",
              file=sys.stderr)
        return 2
    if mode not in backend.capabilities:
        print(f"backend {backend.backend_id} does not support mode={mode!r}; "
              f"capabilities={sorted(backend.capabilities)}",
              file=sys.stderr)
        return 2

    # Prompt template: anchor.yaml for t2i, edit.yaml for edit.
    config = load_anchor() if mode == "text2image" else load_edit()
    # Detail-phrase config: vision QC always describes the full detail
    # (it cares about visible features), so use the anchor's
    # detail_phrases vocabulary regardless of mode.
    qc_cfg = load_anchor()

    # Source provider (edit mode only).
    source_provider = None
    if mode == "edit":
        if not args.source:
            print("--mode=edit requires --source", file=sys.stderr)
            return 2
        if args.source == "backend":
            if not args.source_backend:
                print("--source=backend requires --source-backend",
                      file=sys.stderr)
                return 2
            from .sources.backend import BackendSourceProvider
            src_backend = make_backend(args.source_backend)
            source_provider = BackendSourceProvider(
                src_backend,
                t2i_prompt_fn=lambda s: load_anchor().build_prompt(s),
                seed_fn=stable_seed_for,
            )
        elif args.source == "fixed-file":
            if not args.source_file:
                print("--source=fixed-file requires --source-file <path>",
                      file=sys.stderr)
                return 2
            source_provider = _build_source_provider(
                "fixed-file", Path(args.source_file),
            )
        else:
            source_root = Path(args.source_root) if args.source_root else None
            source_provider = _build_source_provider(args.source, source_root)

    specs = load_variant_specs()
    if args.only:
        specs = [s for s in specs if s["urlHash"] == args.only]
        if not specs:
            print(f"no variant {args.only}", file=sys.stderr)
            return 2
    if args.style:
        wanted = {s.strip() for s in args.style.split(",") if s.strip()}
        specs = [s for s in specs if s["style"] in wanted]
        if not specs:
            print(f"no variants matching style={args.style!r}", file=sys.stderr)
            return 2
    if args.category:
        wanted = {c.strip() for c in args.category.split(",") if c.strip()}
        specs = [s for s in specs if s["categorySlug"] in wanted]
        if not specs:
            print(f"no variants matching category={args.category!r}",
                  file=sys.stderr)
            return 2
    if args.exclude:
        excl = {h.strip() for h in args.exclude.split(",") if h.strip()}
        specs = [s for s in specs if s["urlHash"] not in excl]
        if not specs:
            print(f"all specs filtered out by --exclude={args.exclude!r}",
                  file=sys.stderr)
            return 2

    if not args.dry and not api_key:
        print("GEMINI_API_KEY not set (needed for QC vision check)",
              file=sys.stderr)
        return 2

    writer = ManifestWriter()
    latest = writer.latest_by_url_hash() if not args.dry else {}
    print(
        f"[pipeline] backend={backend.backend_id} mode={mode} "
        f"source={source_provider.provider_id if source_provider else '-'} "
        f"prompt_hash={config.prompt_hash} variants={len(specs)} "
        f"manifest={writer.path.relative_to(REPO)}",
        flush=True,
    )

    if args.resume:
        specs = [s for s in specs if s["urlHash"] not in latest
                 or latest[s["urlHash"]].qc_status != "ok"]
        print(f"[pipeline] resume — {len(specs)} variants remain", flush=True)

    counts = {"ok": 0, "manual_review": 0, "backend_failed": 0, "dry": 0}
    for i, spec in enumerate(specs):
        if args.limit and (counts["ok"] + counts["manual_review"]) >= args.limit:
            print(f"[pipeline] hit --limit {args.limit}, stopping", flush=True)
            break
        tag = f"{spec['urlHash']} ({spec['categorySlug']}/{spec['style']}/{spec['color']})"
        t0 = time.time()
        # Snapshot tag = backend_id + prompt_hash so different backends'
        # outputs at the same prompt template don't collide in the
        # history tree (e.g. Gemini-anchor vs Qwen-anchor at the same
        # urlHash both produce prompt_hash=8d29d50c5ea70dea otherwise).
        prior = latest.get(spec["urlHash"])
        prior_tag = (
            f"{prior.backend_id}__{prior.prompt_hash}" if prior else None
        )
        status = generate_one(
            spec, backend=backend, config=config, writer=writer,
            api_key=api_key, dry=args.dry,
            mode=mode, source_provider=source_provider,
            qc_detail_phrase_config=qc_cfg,
            latest_tag=prior_tag,
            output_root=(Path(args.output_root) if args.output_root else None),
        )
        elapsed = time.time() - t0
        counts[status] = counts.get(status, 0) + 1
        print(
            f"[{i + 1:>3}/{len(specs)}] {tag} → {status} ({elapsed:.1f}s)",
            flush=True,
        )

    print(f"\n[pipeline] done: {counts}")
    return 0 if counts.get("backend_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
