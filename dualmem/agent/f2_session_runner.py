"""Family-2 task runner — Long-Horizon Online IC Injection (advisor's final design).

Runs ONE Family-2 task = a linear spine of D sessions sharing ONE accumulating
memory bank. Each session is a scripted browse walk that injects one cue online
(the base product image is cue-edited at run time via NanoBanana). After each
session, eval-only RECALL PROBES are run from the current bank state — they are
read-only (retrieve + navigate, no `bank.encode`) so they do not perturb the
spine. There is NO tree and NO bank snapshot/restore: the bank simply persists.

This module does NOT touch `dmvbench_live_runner.run_dmvbench_live` (Family 1
must stay byte-identical) — it imports that runner's stable helpers only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import random as _random

from dualmem.agent.actions import Action, parse_react_response
from dualmem.agent.dmvbench_live_runner import (
    SYSTEM_PROMPT_TEMPLATE, Observation, PRODUCT_RE, _bank_has_slug,
    _build_prompt, _format_memory_payload, _retrieve_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V2 = REPO_ROOT / "data" / "vismem_diag_v2"
BASE_IMG_ROOT = V2 / "images" / "base"
WITH_CUE_IMG_ROOT = V2 / "images" / "with_cue"   # v6: every product carries a baked-in cue
ONLINE_CUE_ROOT = V2 / "images" / "online_cue_v2"    # disk cache of online edits
CUE_REGISTRY = V2 / "cue_registry.json"

# The cue must be visible enough that a visual memory can actually encode it
# (a too-small cue tests nothing) — but still a natural, plausibly-placed
# everyday object that does not obscure the product.
EDIT_PROMPT = (
    "Add a {color} {obj} to this product photograph, {placement}. "
    "The {obj} must be clearly visible and prominent — immediately noticeable "
    "at a glance and occupying a meaningful portion of the frame, not a tiny "
    "detail. At the same time it must look natural and photorealistic: a real "
    "everyday object plausibly placed there, not covering or obscuring the "
    "product itself. Keep the product and the background otherwise unchanged. "
    "Photographic realism, no text, no watermark, no caption overlay."
)

_REGISTRY_CACHE: Optional[dict] = None


def _registry() -> dict:
    """url_hash -> {cat, style, prod_idx}."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        rows = json.loads(CUE_REGISTRY.read_text())["rows"]
        _REGISTRY_CACHE = {r["url_hash"]: r for r in rows}
    return _REGISTRY_CACHE


def base_image_for(url_hash: str) -> Optional[Path]:
    """Resolve a product url_hash to its WITH-CUE image (design v6: every
    product on the storefront carries a unique baked-in cue from
    `cue_registry.json`; the storefront serves these images, so observations
    are with_cue everywhere). The name is kept for back-compat with imports."""
    r = _registry().get(url_hash)
    if not r:
        return None
    p = WITH_CUE_IMG_ROOT / r["cat"] / r["style"] / f"{r['prod_idx']:02d}.png"
    return p if p.exists() else None


def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def _f2_obs(url: str) -> Observation:
    """Family-2 observation: a product page shows its CLEAN base image."""
    m = PRODUCT_RE.match(url)
    if m:
        bp = base_image_for(m.group(1))
        return Observation(url=url, image_path=(str(bp) if bp else ""),
                           title=url, description=url)
    return Observation(url=url, image_path="", title=url, description=url)


@dataclass
class InjectionGT:
    """Ground truth recorded when a cue is injected at run time."""
    cue_id: str
    cue_object: str
    cue_color: str
    product_url_hash: str
    product_url: str
    injection_step: int
    cued_image_path: str


@dataclass
class ProbeResult:
    probe_id: str
    at_session: int
    target_session: int
    reach: int
    target_cue: str
    target_url: Optional[str]
    final_url: Optional[str]
    correct: bool
    vlm_calls: int


@dataclass
class F2TaskResult:
    task_id: str
    system: str
    injections: Dict[int, InjectionGT] = field(default_factory=dict)   # session_idx -> GT
    probes: List[ProbeResult] = field(default_factory=list)
    total_vlm_calls: int = 0


def online_cue_edit(base_path: Path, cue_object: str, cue_color: str,
                    placement: str, out_path: Path, nano_banana) -> Path:
    """Edit a cue onto a base image (cached on disk). Returns the cued image path."""
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = EDIT_PROMPT.format(color=cue_color, obj=cue_object, placement=placement)
    with open(base_path, "rb") as f:
        ref = f.read()
    part = nano_banana._types.Part.from_bytes(data=ref, mime_type="image/png")
    img = nano_banana._call([prompt, part])
    img.convert("RGB").save(out_path)
    return out_path


def _rel(url: str, base_url: str) -> str:
    if base_url in url:
        url = url.split(base_url, 1)[-1] or "/"
    return url.split("?")[0]


def _cue_phrase(cue_id: str) -> str:
    """'wool scarf::red' -> 'red wool scarf'."""
    obj, _, col = cue_id.partition("::")
    return f"{col} {obj}" if col else obj


def _build_walk(task_id: str, session_idx: int, shopping_list: list,
                max_steps: int) -> list:
    """Deterministic scripted browse walk for one session: `max_steps` distinct
    product url_hashes drawn from the session's shopping-list categories."""
    rng = _random.Random(f"f2walk.{task_id}.{session_idx}")
    by_cat: dict = {}
    for uh, r in _registry().items():
        by_cat.setdefault(r["cat"], []).append(uh)
    pool: list = []
    for cat in shopping_list:
        pool += by_cat.get(cat, [])
    rng.shuffle(pool)
    return pool[:max_steps]


def _exec_action(page, action: Action, base_url: str, log: list):
    """Execute one ReAct action on the Playwright page.

    Returns the page (possibly a NEW one if the old crashed). Long-horizon
    tasks (J=10/15/50 × hundreds of probes) eventually trigger Chromium
    page crashes ("Page crashed"); without recovery, every subsequent
    `page.goto` fails with the same error and all remaining probes in the
    task are scored wrong. We catch crash/detach errors and spawn a fresh
    page from the same browser context so the task continues normally.

    Callers MUST reassign: `page = _exec_action(page, ...)`."""
    try:
        if action.type in ("navigate", "click_product") and action.url:
            page.goto(base_url + action.url, wait_until="domcontentloaded",
                      timeout=30000)
        elif action.type == "click_index" and action.index is not None:
            # Only the MAIN content grid — the RecentlyViewed widget and header
            # nav sit OUTSIDE <main> and otherwise pollute the index list with
            # stale /product/ links (the cause of the agent re-landing on the
            # same product). On a category page the grid holds collection tiles;
            # on a collection/search page it holds products — accept both so the
            # agent can drill category -> collection -> product.
            hrefs = page.eval_on_selector_all(
                "main a[href^='/product/'], main a[href^='/collection/']",
                "els => Array.from(new Set(els.map(e => e.getAttribute('href'))))")
            if 0 <= action.index < len(hrefs):
                page.goto(base_url + hrefs[action.index],
                          wait_until="domcontentloaded", timeout=30000)
        elif action.type == "add_to_wishlist":
            if action.url_hash and f"/product/{action.url_hash}" not in page.url:
                page.goto(base_url + f"/product/{action.url_hash}",
                          wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        msg = str(e).lower()
        log.append(f"    [exec error: {e}]")
        # Page-level fatal crash → spawn a fresh page from the same browser
        # context so the next probe step can proceed. Without this, the page
        # stays dead for the rest of the task and every later probe is X.
        if ("page crashed" in msg or "frame was detached" in msg
                or "target page" in msg or "target closed" in msg):
            try:
                new_page = page.context.new_page()
                log.append("    [page recovered: spawned new page]")
                try:
                    page.close()
                except Exception:
                    pass
                return new_page
            except Exception as e2:
                log.append(f"    [page recovery FAILED: {e2}]")
    return page


# ---------------------------------------------------------------------------
# Spine: scripted browse walk (encoding, VLM-free) — injects one cue.
# ---------------------------------------------------------------------------

def _run_session_walk(task, sess, system, nano_banana, log: list) -> Optional[InjectionGT]:
    """Scripted browse walk for one session. Encodes products into the (shared,
    persistent) bank; injects the session's single cue at its injection_step."""
    inj = sess.injection
    walk = _build_walk(task.task_id, sess.session_idx, sess.shopping_list,
                       task.max_steps_per_session)
    caption_fn = getattr(system, "caption_fn", None)
    n_before = len(system.bank.entries())
    gt: Optional[InjectionGT] = None
    for step, uh in enumerate(walk, start=1):
        base_img = base_image_for(uh)
        img_path = str(base_img) if base_img else ""
        cat = _registry().get(uh, {}).get("cat", "item")
        injected_now = False
        if gt is None and step >= inj.injection_step and base_img is not None:
            out = ONLINE_CUE_ROOT / f"{_safe(inj.cue_id)}__{uh}.png"
            try:
                cued = online_cue_edit(base_img, inj.cue_object, inj.cue_color,
                                       inj.cue_placement, out, nano_banana)
                img_path = str(cued)
                gt = InjectionGT(
                    cue_id=inj.cue_id, cue_object=inj.cue_object,
                    cue_color=inj.cue_color, product_url_hash=uh,
                    product_url=f"/product/{uh}", injection_step=step,
                    cued_image_path=str(cued))
                injected_now = True
                log.append(f"  [S{sess.session_idx} walk {step}] INJECTED "
                           f"{inj.cue_id} onto /product/{uh}")
            except Exception as e:
                log.append(f"  [S{sess.session_idx} walk {step}] edit failed: {e}")
        # Encode unless already banked — BUT the injection target must be
        # (re-)encoded with the CUED image even if a base entry exists from an
        # earlier session, else the cue is invisible to memory.
        if injected_now or not _bank_has_slug(system, uh):
            cap = caption_fn(img_path) if (caption_fn and img_path) else None
            try:
                system.bank.encode(image_path=img_path, slug=uh,
                                   encode_text=f"a {cat.replace('_', ' ')}",
                                   caption=cap)
            except Exception as e:
                log.append(f"  [S{sess.session_idx} walk {step}] encode fail {uh}: {e}")
    added = len(system.bank.entries()) - n_before
    if gt is None:
        log.append(f"  [S{sess.session_idx}] WARNING: injection never placed")
    log.append(f"  [S{sess.session_idx}] walk: {len(walk)} products, bank +{added}, "
               f"size={len(system.bank.entries())}")
    return gt


# ---------------------------------------------------------------------------
# Recall probe: eval-only, read-only side-run scored by url_match.
# ---------------------------------------------------------------------------

def _run_recall_probe(probe, system, vlm, target_url: Optional[str], *,
                      base_url: str, playwright_page, recall_steps: int,
                      log: list):
    """One eval-only recall probe. The agent navigates from `/` to the product
    carrying `probe.target_cue_id`; scored by url_match. Does NOT mutate the bank.
    v6: the probe carries the (object, color) directly (every visited product
    has a baked-in cue), so the cue phrase is built from those structured fields.

    Returns (ProbeResult, playwright_page) — `playwright_page` may be a NEW page
    if the old one crashed mid-probe and was recovered. The caller MUST update
    its reference: `pr, page = _run_recall_probe(...)`, otherwise subsequent
    probes will keep hitting the dead page."""
    obj = getattr(probe, "target_cue_object", None)
    col = getattr(probe, "target_cue_color", None)
    cue_phrase = (f"{col} {obj}" if (obj and col) else _cue_phrase(probe.target_cue_id))
    log.append(f"  --- PROBE {probe.probe_id} reach-{probe.reach} "
               f"cue={probe.target_cue_id} gt={target_url} ---")
    # Recover from a page that died at the end of the PREVIOUS probe before
    # we try anything with it. Without this, the initial goto-/ fails and
    # every recall_step after also fails on the dead page.
    if playwright_page is not None:
        try:
            _ = playwright_page.url       # cheap liveness check
        except Exception as e:
            log.append(f"    [page was dead at probe start: {e}]")
            try:
                playwright_page = playwright_page.context.new_page()
                log.append("    [page recovered at probe start]")
            except Exception as e2:
                log.append(f"    [page recovery failed: {e2}]")
                playwright_page = None
    if playwright_page is not None:
        try:
            playwright_page.goto(base_url + "/", wait_until="domcontentloaded",
                                 timeout=30000)
        except Exception as e:
            log.append(f"    [goto / fail: {e}]")
            # Goto failure can itself indicate a crashed page; try recovering.
            msg = str(e).lower()
            if ("page crashed" in msg or "target page" in msg
                    or "target closed" in msg):
                try:
                    playwright_page = playwright_page.context.new_page()
                    log.append("    [page recovered after / goto fail]")
                    playwright_page.goto(base_url + "/",
                                         wait_until="domcontentloaded", timeout=30000)
                except Exception as e2:
                    log.append(f"    [page re-init failed: {e2}]")
    history = [{"role": "user",
                "text": f"Earlier you saw a product that had a {cue_phrase} on "
                        f"it. Take me back to that exact product."}]
    final_url = None
    vlm_calls = 0
    for rstep in range(1, recall_steps + 1):
        # Read current URL; if the page died between steps, recover.
        try:
            cur = _rel(playwright_page.url, base_url) if playwright_page else "/"
        except Exception as e:
            log.append(f"    [page.url failed mid-loop: {e}]")
            try:
                playwright_page = playwright_page.context.new_page()
                playwright_page.goto(base_url + "/", wait_until="domcontentloaded",
                                     timeout=30000)
                log.append("    [page recovered mid-loop]")
                cur = "/"
            except Exception as e2:
                log.append(f"    [mid-loop recovery failed: {e2}]")
                cur = "/"
        obs = _f2_obs(cur)
        hits, top_k = _retrieve_step(
            system, f"product with {cue_phrase} CURRENT: {obs.url}", k=5)
        mem_text, mem_images = _format_memory_payload(system, hits)
        prompt = _build_prompt(
            subtask_description=f"Recall: go to the product with the {cue_phrase}.",
            history_messages=history, memory_text=mem_text,
            current_obs=obs, step=rstep, cap=recall_steps)
        try:
            a_text = vlm.generate_freeform(
                system_prompt=SYSTEM_PROMPT_TEMPLATE, user_text=prompt,
                primary_image=(obs.image_path or None),
                extra_images=(mem_images or None), max_tokens=512)
        except Exception as e:
            a_text = f"Thought: vlm error\nAction: done   # {e}"
        vlm_calls += 1
        action = parse_react_response(a_text)
        log.append(f"    R{rstep}/{recall_steps} obs={obs.url} retrieved={top_k}")
        for ln in a_text.strip().splitlines():
            log.append(f"    | {ln}")
        log.append(f"    -> {action.type} {action.url or ''}")
        if playwright_page is not None:
            playwright_page = _exec_action(playwright_page, action, base_url, log)
        # Guard cur2 read too — if _exec_action just recovered the page,
        # reading .url should work; if it returned None, fall back to obs.url.
        try:
            cur2 = _rel(playwright_page.url, base_url) if playwright_page else obs.url
        except Exception as e:
            log.append(f"    [page.url read failed: {e}]")
            cur2 = obs.url
        if PRODUCT_RE.match(cur2):
            final_url = cur2
        history.append({"role": "agent", "text": a_text})
        if action.type in ("add_to_wishlist", "done"):
            break
    correct = (final_url is not None and target_url is not None
               and final_url == target_url)
    log.append(f"    probe_final={final_url} gt={target_url} correct={correct}")
    return ProbeResult(
        probe_id=probe.probe_id, at_session=probe.at_session,
        target_session=probe.target_session, reach=probe.reach,
        target_cue=probe.target_cue_id, target_url=target_url,
        final_url=final_url, correct=correct, vlm_calls=vlm_calls), playwright_page


# ---------------------------------------------------------------------------
# Whole-task driver: linear spine + probes.
# ---------------------------------------------------------------------------

def _replay_trajectory(traj, system, log: list) -> None:
    """Encode a recorded encoding-session trajectory into the (persistent) bank.
    Design v6: every visited product is encoded with its WITH-CUE image (the
    cue is baked into the storefront), so there is no per-cue special case."""
    caption_fn = getattr(system, "caption_fn", None)
    n_before = len(system.bank.entries())
    # Tell session-aware adapters (e.g. WorldMM-lite multi-grain) which
    # session is about to be encoded. No-op for plain banks.
    setter = getattr(system.bank, "set_session", None)
    if callable(setter):
        setter(traj.session_idx)

    for uh, img_path in traj.visited:
        if _bank_has_slug(system, uh):
            continue
        cat = _registry().get(uh, {}).get("cat", "item")
        cap = caption_fn(img_path) if (caption_fn and img_path) else None
        try:
            system.bank.encode(image_path=img_path, slug=uh,
                               encode_text=f"a {cat.replace('_', ' ')}", caption=cap)
        except Exception as e:
            log.append(f"  [replay] encode fail {uh}: {e}")
    added = len(system.bank.entries()) - n_before
    log.append(f"  [S{traj.session_idx}] replay: {len(traj.visited)} product views, "
               f"bank +{added} -> size={len(system.bank.entries())}")


def run_f2_task(
    task,                       # RolloutTreeTask (linear spine + probes)
    system,                     # a fresh MemorySystem
    vlm,
    *,
    trajectories,               # list[EncodeTrajectory] — one per session, pre-generated
    base_url: str = "http://localhost:3000",
    playwright_page=None,
    recall_steps: int = 8,
    log_path: Optional[Path] = None,
    verbose: bool = True,
    mc_probes: int = 0,         # > 0 → Monte Carlo sampling at probe-build time
    mc_seed: str = "f2.mc.default",
) -> F2TaskResult:
    """Run one whole Family-2 task for one memory system: replay the recorded
    encoding trajectories into the bank, then run the recall probes.

    `mc_probes > 0` switches probe construction from exhaustive
    (`k·N(N-1)/2`) to Monte Carlo (`~mc_probes × (N-1)`), enabling
    long-horizon coverage at much lower cost. See
    `tasks/generators/f2_online_ic.build_probes_from_trajectories` for the
    sampling contract."""
    log: List[str] = []

    def _flush():
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log))

    log.append(f"=== F2 task={task.task_id} system={system.name} "
               f"D={task.n_sessions} ===")
    system.reset()
    result = F2TaskResult(task_id=task.task_id, system=system.name)
    traj_by_idx = {t.session_idx: t for t in trajectories}

    # v6: fill probes at run time from the recorded encoding trajectories
    # (one probe per (at_session j, target_session, viewed product)).
    if not task.probes:
        from tasks.generators.f2_online_ic import build_probes_from_trajectories
        traj_visits = {j: traj_by_idx[j].visited for j in traj_by_idx}
        task.probes = build_probes_from_trajectories(
            task, traj_visits, _registry(),
            mc_probes=mc_probes, mc_seed=mc_seed)
        log.append(f"  built {len(task.probes)} probes from trajectories"
                   + (f"  (MC, {mc_probes}/reach, seed={mc_seed})"
                      if mc_probes > 0 else "  (exhaustive)"))

    for j in range(task.n_sessions):
        traj = traj_by_idx.get(j)
        if traj is None:
            log.append(f"  [S{j}] WARNING: no trajectory — skipped")
            _flush()
            continue
        _replay_trajectory(traj, system, log)
        _flush()
        for probe in task.probes_at(j):
            target_url = f"/product/{probe.target_url_hash}"
            pr, playwright_page = _run_recall_probe(
                probe, system, vlm, target_url,
                base_url=base_url, playwright_page=playwright_page,
                recall_steps=recall_steps, log=log)
            result.probes.append(pr)
            result.total_vlm_calls += pr.vlm_calls
            _flush()
            if verbose:
                print(f"    {task.task_id} {probe.probe_id} reach-{probe.reach} "
                      f"{'OK' if pr.correct else 'X'}", flush=True)

    log.append(f"  total_vlm_calls={result.total_vlm_calls}")
    _flush()
    return result
