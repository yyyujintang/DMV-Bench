"""Incidental-Cue task runner.

One task is a linear chain of D sessions that share a single accumulating
memory bank. The encoding trajectories are produced separately (see
`f2_encode_agent`) and are replayed here into whichever memory system is under
test, so every system sees an identical observation stream.

After each session is replayed, eval-only RECALL PROBES run from the current
bank state. Probes are read-only -- they retrieve and navigate but never call
`bank.encode` -- so they cannot perturb the chain. There is no branching and no
bank snapshot/restore: the bank simply persists across the chain.

Cues are baked into the catalog images at build time (the storefront serves
`images/with_cue/...`), so nothing is edited at run time.

This module imports stable helpers from `dualmem.agent.prompting` (the shared
system prompt, observation type and retrieval/prompt formatting).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dualmem.agent.actions import Action, parse_react_response
from dualmem.agent.prompting import (
    SYSTEM_PROMPT_TEMPLATE, Observation, PRODUCT_RE, _bank_has_slug,
    _build_prompt, _format_memory_payload, _retrieve_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V2 = REPO_ROOT / "data" / "vismem_diag_v2"
WITH_CUE_IMG_ROOT = V2 / "images" / "with_cue"
CUE_REGISTRY = V2 / "cue_registry.json"

_REGISTRY_CACHE: Optional[dict] = None


def _registry() -> dict:
    """url_hash -> {cat, style, prod_idx}."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        rows = json.loads(CUE_REGISTRY.read_text())["rows"]
        _REGISTRY_CACHE = {r["url_hash"]: r for r in rows}
    return _REGISTRY_CACHE


def image_for(url_hash: str) -> Optional[Path]:
    """Resolve a product url_hash to the image the storefront serves for it.

    Every product carries a unique cue baked in at build time, so this is
    always the `with_cue` render -- byte-identical to the photo on the page."""
    r = _registry().get(url_hash)
    if not r:
        return None
    p = WITH_CUE_IMG_ROOT / r["cat"] / r["style"] / f"{r['prod_idx']:02d}.png"
    return p if p.exists() else None


# Kept as an alias: `f2_encode_agent` and external scripts import this name.
base_image_for = image_for


def _f2_obs(url: str) -> Observation:
    """The agent's view of the current page.

    `title` and `description` are deliberately set to the URL rather than to
    scraped DOM text: the harness does not read the page's DOM, so the only
    textual signal about the current page is its address. This is uniform
    across every memory system, so it cannot bias a comparison between them.
    On a product page the agent additionally receives the product image; other
    page types carry no image.
    """
    m = PRODUCT_RE.match(url)
    if m:
        bp = image_for(m.group(1))
        return Observation(url=url, image_path=(str(bp) if bp else ""),
                           title=url, description=url)
    return Observation(url=url, image_path="", title=url, description=url)


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
    probes: List[ProbeResult] = field(default_factory=list)
    total_vlm_calls: int = 0


def _rel(url: str, base_url: str) -> str:
    if base_url in url:
        url = url.split(base_url, 1)[-1] or "/"
    return url.split("?")[0]


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
# Recall probe: eval-only, read-only side-run scored by url_match.
# ---------------------------------------------------------------------------

def _run_recall_probe(probe, system, vlm, target_url: Optional[str], *,
                      base_url: str, playwright_page, recall_steps: int,
                      log: list):
    """One eval-only recall probe. The agent navigates from `/` to the product
    carrying `probe.target_cue_id`; scored by url_match. Does NOT mutate the
    bank. The probe carries the cue's (object, colour) as structured fields, so
    the phrase shown to the agent is built directly from those.

    Returns (ProbeResult, playwright_page) — `playwright_page` may be a NEW page
    if the old one crashed mid-probe and was recovered. The caller MUST update
    its reference: `pr, page = _run_recall_probe(...)`, otherwise subsequent
    probes will keep hitting the dead page."""
    cue_phrase = f"{probe.target_cue_color} {probe.target_cue_object}"
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
# Whole-task driver: replay the chain, probe after every session.
# ---------------------------------------------------------------------------

def _replay_trajectory(traj, system, log: list) -> None:
    """Encode a recorded encoding-session trajectory into the (persistent)
    bank. Every visited product is encoded with the image the storefront served
    for it, cue included, so there is no per-cue special case."""
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
    task,                       # ChainTask: the D-session chain + its probes
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
    """Run one whole task for one memory system: replay the recorded encoding
    trajectories into the bank, then run the recall probes.

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

    log.append(f"=== task={task.task_id} system={system.name} "
               f"D={task.n_sessions} ===")
    system.reset()
    result = F2TaskResult(task_id=task.task_id, system=system.name)
    traj_by_idx = {t.session_idx: t for t in trajectories}

    # Probes are filled at run time from the recorded trajectories -- which
    # products the agent saw is autonomous, so they cannot be fixed in advance.
    # One probe per (probe session j, target session, product viewed there).
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
