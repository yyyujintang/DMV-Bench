"""F2-side MemorySystem wrappers around the Trial-based external adapters.

The external baselines in `dualmem/baselines/external/` (WorldMM, M2A, MMA)
were authored against the v1 Trial-based interface (`encode(Trial)` /
`retrieve(query_text) -> Optional[Trial]`). The F2 pipeline uses a
MemorySystem(bank, retriever, injector) interface (see `dualmem/systems/`).

This module bridges them: a thin `_AdapterBank` that forwards `encode(...)`
through a stub Trial; a thin `_AdapterRetriever` that maps the adapter's
returned Trial.anchor_slug back to a MemoryEntry from the bank; and a
`make_external_system(name)` factory used by `dualmem/systems/registry.py`.

External-repo failures degrade gracefully: each adapter's "spirit-of" local
fallback is what runs when the upstream repo isn't on disk.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from dualmem.injection import CaptionOnlyInject, ImageTextInject
from dualmem.memory.entry import MemoryEntry
from dualmem.systems.base import MemorySystem
from dualmem.types import TaskCell, Trial


def _stub_trial(slug: str, image_path: str, encode_text: str = "",
                filler_steps: int = 0) -> Trial:
    """Minimal Trial — adapters only read `anchor_slug`, `anchor_path`,
    `encode_text`, and (sometimes) `filler_steps` during encode."""
    cell = TaskCell(category="external", variant="ext", grain_tier=1,
                    leakage_level=0, mechanism="descriptive",
                    ceiling="full_pipeline", seed=0)
    return Trial(
        cell=cell, anchor_slug=slug, anchor_path=image_path,
        candidate_paths=[image_path], candidate_slugs=[slug], correct_index=0,
        encode_text=encode_text, recall_text=encode_text,
        filler_steps=filler_steps,
    )


class _AdapterBank:
    """Forwards `encode/reset` to a Trial-based adapter; keeps a local list of
    MemoryEntry so `entries()` / `find_by_slug()` satisfy the MemoryBank
    Protocol used by F2's retrievers and injectors."""

    def __init__(self, adapter, name: str):
        self.adapter = adapter
        self.name = name
        self._entries: List[MemoryEntry] = []
        self._by_slug: Dict[str, MemoryEntry] = {}

    def reset(self) -> None:
        try:
            self.adapter.reset()
        except Exception:
            pass
        self._entries = []
        self._by_slug = {}

    def set_session(self, s: int) -> None:
        """Hook used by F2's `_replay_trajectory` to tell the adapter which
        session is about to be encoded — enables session-aware temporal
        indexes (e.g. WorldMM-lite's multi-grain `set_session`). Silently
        no-ops if the underlying adapter doesn't expose `set_session`."""
        fn = getattr(self.adapter, "set_session", None)
        if callable(fn):
            try:
                fn(s)
            except Exception:
                pass

    def encode(self, image_path: str, slug: str, *,
               encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry:
        if slug in self._by_slug:
            return self._by_slug[slug]
        text = encode_text or (caption or "")
        try:
            self.adapter.encode(_stub_trial(slug, image_path, text,
                                            filler_steps=len(self._entries)))
        except Exception:
            # External-adapter failures (missing repo, VLM hiccup) must not
            # crash the run — the entry is still stored for fallback retrieval.
            pass
        entry = MemoryEntry(
            slug=slug, image_path=image_path,
            encode_text=encode_text, caption=caption,
            encoder_name=f"ext:{self.name}",
        )
        self._entries.append(entry)
        self._by_slug[slug] = entry
        return entry

    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def find_by_slug(self, slug: str) -> Optional[MemoryEntry]:
        return self._by_slug.get(slug)


class _AdapterRetriever:
    """Forwards `retrieve(query, entries, k)` to the adapter's
    `retrieve(query_text)`; maps the returned Trial.anchor_slug back to the
    bank's MemoryEntry. Pads to k with other entries (preserving the adapter
    pick at rank 0) so callers expecting top-k get a non-empty list."""

    def __init__(self, adapter, bank: _AdapterBank, name: str):
        self.adapter = adapter
        self.bank = bank
        self.name = name

    def retrieve(self, query, entries: List[MemoryEntry], k: int = 1) -> List[MemoryEntry]:
        """Real top-k via the adapter's own ranked scoring when available
        (`retrieve_topk(query, k) -> List[slug]`); only falls back to
        top-1 + insertion-order padding for adapters that don't expose
        that. The earlier top-1-only path caused systematic recall-bank
        failure at long reach: padding was junk."""
        qtext = (getattr(query, "recall_text", None)
                 or getattr(query, "text", None)
                 or str(query))
        # Preferred path: ranked top-k from the adapter's full scoring.
        topk_fn = getattr(self.adapter, "retrieve_topk", None)
        if callable(topk_fn):
            try:
                slugs = topk_fn(qtext, k) or []
            except Exception:
                slugs = []
            out: List[MemoryEntry] = []
            for s in slugs:
                hit = self.bank.find_by_slug(s)
                if hit is not None and hit.slug not in {x.slug for x in out}:
                    out.append(hit)
                if len(out) >= k:
                    break
            if out:
                # pad if adapter returned <k, with remaining entries (rare)
                for e in entries:
                    if e.slug in {x.slug for x in out}:
                        continue
                    out.append(e)
                    if len(out) >= k:
                        break
                return out[:k]
        # Legacy fallback: adapter only exposes single-best retrieve.
        try:
            tr = self.adapter.retrieve(qtext)
        except Exception:
            tr = None
        out = []
        if tr is not None:
            hit = self.bank.find_by_slug(tr.anchor_slug)
            if hit is not None:
                out.append(hit)
        for e in entries:
            if e.slug in {x.slug for x in out}:
                continue
            out.append(e)
            if len(out) >= k:
                break
        return out[:k]


# Adapters default to a local 7B Qwen (each adapter would load its own copy
# → 3× ~14GB GPU = OOM on a single device, slow startup). We construct them
# directly with the same hosted-API VLM that drives the F2 agent (Gemini
# Flash by default), keeping the comparison fair (same VLM everywhere) AND
# the smoke / large-scale runs fast.
_DEFAULT_EXT_VLM = "gemini-flash"


def _build_adapter(name: str, vlm_backend: str):
    """Construct the right Adapter with our shared VLM backend wired in.

    Names accepted (after honest renaming — see
    `doc/multimodal_baseline_adaptation.md` §Residual-concentration):
      "WorldMM-lite" / "WorldMM"   -> WorldMMAdapter (we keep tri-modular
                                        text store + cue-router; we drop
                                        HippoRAG/PPR, multi-granularity time
                                        index, VLM2Vec, multi-round agent)
      "M2A-lite"     / "M2A"       -> M2AAdapter (we keep dual-layer text
                                        store; we drop BM25 + siglip2 hybrid
                                        retrieval)
      "MMA-lite"     / "MMA"       -> MMAAdapter (faithful R(m|q) score —
                                        named "-lite" for naming consistency
                                        only; the formula is the paper's.)
    """
    key = name.replace("-lite", "")
    if key == "WorldMM":
        from dualmem.baselines.external.worldmm import WorldMMAdapter
        return WorldMMAdapter(retriever_vlm=vlm_backend, responder_vlm=vlm_backend)
    if key == "M2A":
        from dualmem.baselines.external.m2a import M2AAdapter
        return M2AAdapter(vlm_backend=vlm_backend)
    if key == "MMA":
        from dualmem.baselines.external.mma import MMAAdapter
        return MMAAdapter(vlm_backend=vlm_backend)
    raise KeyError(f"Unknown external baseline: {name!r}")


# Per-baseline injector — chosen to MATCH each paper's original protocol
# (the encode/retrieve/inject decomposition is DualMem's contribution and is
# not used to ablate external methods; each external baseline runs as its
# paper does).
#   M2A      — paper emits a free-text answer; image is NOT re-presented at
#              recall  → CaptionOnlyInject (text only).
#   WorldMM  — paper's responder loop sees module text + retrieved frames
#              → ImageTextInject.
#   MMA      — multimodal memory agent; retrieved items (text + image) feed
#              the answerer after reliability re-scoring  → ImageTextInject.
_PAPER_INJECTOR = {
    "M2A-lite":      ("text",  True),    # (injector_kind, needs_F2_caption)
    "WorldMM-lite":  ("image", False),
    "MMA-lite":      ("image", False),
}


def make_external_system(name: str, vlm_backend: str = _DEFAULT_EXT_VLM,
                         vlm_caption_fn=None) -> MemorySystem:
    """Build a F2 MemorySystem matching the named external baseline's
    ORIGINAL paper protocol. Each baseline corresponds to one (encode,
    retrieve, inject) configuration — paper-faithful, no per-baseline
    ablations (that would confound the external comparison)."""
    kind, needs_cap = _PAPER_INJECTOR.get(name, ("image", False))
    adapter = _build_adapter(name, vlm_backend)
    bank = _AdapterBank(adapter, name)
    retriever = _AdapterRetriever(adapter, bank, name)
    return MemorySystem(
        name=name, bank=bank, retriever=retriever,
        injector=(CaptionOnlyInject() if kind == "text" else ImageTextInject()),
        caption_fn=(vlm_caption_fn if needs_cap else None),
    )
