"""Encoding-trajectory generator -- a real ReAct shopping agent.

ONE SESSION = ONE short WebArena-style shopping task: browse a single category
for ~22-28 steps and open as many distinct products as possible. Sessions are
deliberately SHORT; the long horizon comes from CHAINING many of them, not from
long sessions. Within a session the agent sees its whole history.

The encoding agent uses NO memory, so a session trajectory depends only on
(seed, session_idx). It is generated ONCE, recorded, and replayed into every
memory system under test, which is what makes the comparison paired and lets a
J=5 and a J=15 chain on the same seed share their early sessions exactly.

Cues are baked into the catalogue images at build time, so the agent simply
sees them; nothing is injected at run time and the agent is never told they
exist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from dualmem.agent.actions import parse_react_response
from dualmem.agent.prompting import (
    SYSTEM_PROMPT_TEMPLATE, PRODUCT_RE, _build_prompt,
)
from dualmem.agent.f2_session_runner import _exec_action, _f2_obs, _rel, _registry

# A WebArena-style comparison-shopping task: browse widely before deciding, so
# each session encodes ~12 distinct products (a bigger memory bank). Product
# pages have a "More from this collection" grid, so the agent can page from one
# product to the next within a collection without going back to the category.
SESSION_TASK = (
    "You are comparison-shopping for a {cat} on a furniture store website. "
    "Your goal is BREADTH OF EXPLORATION, not picking a winner: visit AT "
    "LEAST 3 different {cat} collections (styles), and within each open "
    "SEVERAL individual products (a product page has a 'More from this "
    "collection' grid you can use to move to the next product). "
    "\n\nIMPORTANT BROWSING RULES:\n"
    "  - Keep navigating to NEW products until you reach the step cap. "
    "Repeating the same product is a wasted step.\n"
    "  - DO NOT emit add_to_wishlist or done — those terminate the browse "
    "early and the step is wasted. Even if a product looks great, KEEP "
    "BROWSING other products and styles.\n"
    "  - From a product page, use navigate(\"/collection/{cat}-<style>\") "
    "to jump to a different style, or click_index(N) on the in-page grid "
    "to move to the next product card."
)
# IC-safety note: do NOT add any rule that mentions "incidental details" or
# tells the agent to ignore extra objects on product images — DMV-Bench's
# whole point is the INCIDENTAL CUE (IC) signal. The agent must see the
# cued image with NO awareness it is a cue, so memory baselines can be
# scored on whether they spontaneously encoded the cue feature.


@dataclass
class EncodeTrajectory:
    """The recorded encoding trajectory of one session -- replayed into every
    memory system's bank."""
    session_idx: int
    visited: List[Tuple[str, str]] = field(default_factory=list)  # (url_hash, image_path)
    n_steps: int = 0
    vlm_calls: int = 0


def _noun(cat: str) -> str:
    return {"plant_pot": "plant pot", "wall_art": "wall-art piece"}.get(
        cat, cat.replace("_", " "))


def run_encode_session(
    session_spec,                # SessionSpec
    *,
    vlm,
    base_url: str = "http://localhost:3000",
    playwright_page=None,
    log: Optional[list] = None,
) -> EncodeTrajectory:
    """Run ONE short encoding session: a real ReAct agent browses one category
    for `session_spec.n_steps` steps (no memory, whole-session history).
    Records every product page it viewed, with the image the storefront
    served -- which already carries that product's cue."""
    log = log if log is not None else []
    cat = session_spec.shopping_list[0]
    n_steps = session_spec.n_steps
    traj = EncodeTrajectory(session_idx=session_spec.session_idx)
    log.append(f"--- ENCODE session {session_spec.session_idx} | category={cat} | "
               f"n_steps={n_steps} ---")

    if playwright_page is not None:
        try:
            playwright_page.goto(base_url + f"/category/{cat}",
                                 wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            log.append(f"  [goto fail: {e}]")

    history: List[dict] = [{"role": "user",
                            "text": SESSION_TASK.format(cat=_noun(cat))}]
    for step in range(1, n_steps + 1):
        cur = _rel(playwright_page.url, base_url) if playwright_page else "/"
        obs = _f2_obs(cur)
        m = PRODUCT_RE.match(obs.url)
        # The cue is already in the storefront image; `_f2_obs` resolves the
        # URL to that same file, so recording the path is all that is needed.
        if m:
            traj.visited.append((m.group(1), obs.image_path or ""))

        # progress signals (not history → never truncated) — guide the agent to
        # browse widely: many distinct products, >=3 distinct styles.
        seen = {u for u, _ in traj.visited}
        styles = {_registry().get(u, {}).get("style") for u in seen}
        styles.discard(None)
        prompt = _build_prompt(
            subtask_description=(
                f"Comparison-shop for a {_noun(cat)} — open a wide range "
                f"(step {step}/{n_steps}). So far: {len(seen)} distinct products "
                f"viewed across {len(styles)} styles. Aim for >=3 styles and "
                f"~12 products; open NEW products you have not seen yet."),
            history_messages=history, memory_text="",   # encoding agent: NO memory
            current_obs=obs, step=step, cap=n_steps)
        try:
            a_text = vlm.generate_freeform(
                system_prompt=SYSTEM_PROMPT_TEMPLATE, user_text=prompt,
                primary_image=(obs.image_path or None), extra_images=None,
                max_tokens=512)
        except Exception as e:
            a_text = f"Thought: vlm error\nAction: done   # {e}"
        traj.vlm_calls += 1
        action = parse_react_response(a_text)
        # compact action history (standard web-agent practice)
        history.append({"role": "agent",
                        "text": f"{action.type} {action.url or ''}".strip()})
        log.append(f"  step {step}/{n_steps} {obs.url} -> {action.type} "
                   f"{action.url or ''}")

        if action.type in ("done", "add_to_wishlist"):
            # SHORT fixed-budget browse: do not end early — force-navigate
            # back to the category root so the next step lands on a fresh
            # listing. Without this, smaller VLMs (e.g. Qwen2.5-VL-7B) get
            # stuck on the same product re-emitting add_to_wishlist forever.
            history.append({"role": "user",
                            "text": f"Don't terminate — keep browsing other "
                                    f"{_noun(cat)} products and styles."})
            if playwright_page is not None:
                try:
                    playwright_page.goto(base_url + f"/category/{cat}",
                                         wait_until="domcontentloaded",
                                         timeout=15000)
                    log.append(f"  [step {step}] forced-nav -> /category/{cat} "
                               f"(broke {action.type} loop)")
                except Exception as e:
                    log.append(f"  [step {step}] forced-nav failed: {e}")
        elif playwright_page is not None:
            _exec_action(playwright_page, action, base_url, log)

    traj.n_steps = n_steps
    distinct = len({u for u, _ in traj.visited})
    log.append(f"  ENCODE done: {n_steps} steps, {len(traj.visited)} product "
               f"views ({distinct} distinct)")
    return traj


# ---------------------------------------------------------------------------
# Trajectory cache + whole-task generation
# ---------------------------------------------------------------------------

def save_trajectory(traj: EncodeTrajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {"session_idx": traj.session_idx,
         "visited": [list(v) for v in traj.visited],
         "n_steps": traj.n_steps, "vlm_calls": traj.vlm_calls}
    path.write_text(json.dumps(d, indent=2))


def load_trajectory(path: Path) -> EncodeTrajectory:
    d = json.loads(Path(path).read_text())
    return EncodeTrajectory(
        session_idx=d["session_idx"],
        visited=[tuple(v) for v in d["visited"]],
        n_steps=d["n_steps"], vlm_calls=d.get("vlm_calls", 0))


def generate_task_trajectories(
    task,                        # ChainTask
    vlm, *,
    traj_dir: Path,
    base_url: str = "http://localhost:3000",
    playwright_page=None,
    log: Optional[list] = None,
) -> List[EncodeTrajectory]:
    """Run the encoding agent for every session of `task` (baseline-independent),
    caching each trajectory to `traj_dir/<task_id>__s<j>.json`. A cached
    trajectory is reused — so a J=5 task and a later J=20 task with the same
    seeds share their session trajectories."""
    log = log if log is not None else []
    out: List[EncodeTrajectory] = []
    seed = task.metadata.get("seed", 0)
    for sess in task.sessions:
        # cache key = (seed, session_idx) — J-INDEPENDENT, so a J=5 run and a
        # later J=10/J=15 run with the same seed reuse sessions 0..4.
        cache = Path(traj_dir) / f"f2traj_s{seed:04d}_sess{sess.session_idx}.json"
        if cache.exists():
            traj = load_trajectory(cache)
            log.append(f"  [traj] session {sess.session_idx}: cached "
                       f"({len(traj.visited)} views)")
        else:
            traj = run_encode_session(sess, vlm=vlm, base_url=base_url,
                                      playwright_page=playwright_page, log=log)
            save_trajectory(traj, cache)
        out.append(traj)
    return out
