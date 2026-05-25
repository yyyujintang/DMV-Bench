"""Playwright-driven VLM shopping agent.

One agent run = one AgentTask under one (system × ceiling) combo:
    1. Navigate to anchor URL → encode page observation
    2. Navigate filler URLs → encode each
    3. Recall loop:
        a. Capture current page (screenshot + url)
        b. Build VLM prompt (current obs + memory context per ceiling)
        c. VLM emits ReAct response
        d. Parse action; execute via Playwright
        e. Repeat until done() / add_to_wishlist() / step cap
    4. Score: final URL ~ task.expected_url_pattern
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dualmem.agent.actions import Action, parse_react_response
from dualmem.agent.observation import PageObservation
from dualmem.retrieval.base import RetrievalQuery
from dualmem.tasks.spec import AgentTask


# ----------------------------------------------------------------------
# Per-task result row
# ----------------------------------------------------------------------

@dataclass
class AgentRunResult:
    task_id: str
    system: str
    ceiling: str
    n_steps: int
    final_url: str
    correct: bool
    elapsed_ms: int
    transcript: str        # full VLM + action log
    failure_mode: str = "" # "" / "step_cap" / "noop_loop" / "parse_error"


# ----------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a shopping assistant agent navigating a furniture e-commerce site.
You see one page at a time and choose ONE action per turn.

Reply in ReAct format:
  Thought: <one sentence reasoning>
  Action: <action call>

Available actions (use the EXACT syntax — wrong format = no-op):
  navigate("/category/<slug>?tier=N")      — go to a category page
  click_index(N)                           — on a category page, click the N-th product card
                                              (0 = top-left, then left-to-right top-to-bottom).
                                              Indices 0..3 are valid (the grid has 4 items).
  add_to_wishlist                          — on a product detail page, click the wishlist button
                                              (TASK COMPLETION — use this when you are on the
                                              right product page)
  done                                     — finish the run if you are stuck

Tips:
  · On a category page you cannot read the URL hashes; use click_index(N).
  · You succeed by clicking add_to_wishlist on the SAME product you saw earlier (visually).
  · Output one Thought + one Action line, nothing else."""


def _build_recall_prompt(
    obs: PageObservation,
    task: AgentTask,
    memory_text: str,
    n_step: int,
    n_cap: int,
) -> str:
    return f"""Step {n_step}/{n_cap}.

Current page URL: {obs.url}
Current page title: {obs.title}

[Memory context]
{memory_text or "(no memory provided this turn)"}

[Recall instruction from user]
{task.recall_instruction}

The candidates collection page is: {task.recall_collection_url}

Reply with one Thought + one Action.
"""


# ----------------------------------------------------------------------
# Page → observation
# ----------------------------------------------------------------------

PRODUCT_URL_RE = re.compile(r"/product/([0-9a-f]{6,16})$")


def _capture(page, step: int, out_dir: Path) -> PageObservation:
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / f"step_{step:02d}.png"
    page.screenshot(path=str(shot), full_page=False)
    url = page.url
    title = page.title()
    # /product/<hash> detection
    rel = url.split("//", 1)[-1].split("/", 1)[-1] if "//" in url else url
    rel = "/" + rel if not rel.startswith("/") else rel
    m = PRODUCT_URL_RE.search(rel.split("?")[0])
    product_hash = m.group(1) if m else None
    return PageObservation(
        url=rel,
        screenshot_path=str(shot),
        title=title,
        product_url_hash=product_hash,
        step_index=step,
    )


def _capture_collection_grid(page, step: int, out_dir: Path) -> Optional[str]:
    """On a /category/<slug> page, screenshot each product card individually
    at native resolution and stitch them into a 2×2 grid labelled [0]/[1]/[2]/[3].

    The label index matches DOM order (= click_index(N) target). Returns the
    grid path, or None if not a category page or card count ≠ 4.

    Motivation: full-page screenshots at viewport 1280×900 shrink each card
    thumbnail to ~200×200 px, which is too low-res for VLM to distinguish
    near-identical tier-3 variants. The grid preserves native card pixels
    and zooms each to 512×512.
    """
    if "/category/" not in page.url:
        return None
    try:
        cards = page.locator("a[href^='/product/'] div.aspect-square")
        n = cards.count()
    except Exception:
        return None
    if n != 4:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    side = 512
    grid = Image.new("RGB", (side * 2, side * 2), "white")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for i in range(4):
        try:
            buf = cards.nth(i).screenshot()
        except Exception:
            return None
        im = Image.open(io.BytesIO(buf)).convert("RGB")
        im = im.resize((side, side), Image.LANCZOS)
        x, y = (i % 2) * side, (i // 2) * side
        grid.paste(im, (x, y))
        # White-bordered black label tile.
        draw.rectangle([x + 8, y + 8, x + 100, y + 80], fill="black")
        draw.text((x + 22, y + 14), f"[{i}]", fill="white", font=font)
    path = out_dir / f"step_{step:02d}_grid.png"
    grid.save(path)
    return str(path)


# ----------------------------------------------------------------------
# Memory hook helpers
# ----------------------------------------------------------------------

def _encode_observation(system, obs: PageObservation, encode_text: str):
    """Translate a page obs into the bank's encode() contract.

    Slug = URL-hash for product pages, "page_<step>" for non-product pages.
    Caption: deferred to system.caption_fn(image_path) if the system needs it.
    """
    slug = obs.product_url_hash or f"page_{obs.step_index:02d}"
    caption = None
    if system.caption_fn is not None and obs.screenshot_path:
        caption = system.caption_fn(obs.screenshot_path)
    system.bank.encode(
        image_path=obs.screenshot_path,
        slug=slug,
        encode_text=encode_text,
        caption=caption,
    )


def _memory_payload_for_ceiling(system, task: AgentTask, ceiling: str) -> dict:
    """Return {'text': str, 'images': [paths]} for the current step's memory context.

    - ceiling='perception': bypass memory entirely → inject the anchor's screenshot
    - ceiling='oracle_retrieval': bypass retrieval → inject the anchor as memory entry
    - ceiling='full_pipeline': retrieve from bank using recall_instruction
    """
    if ceiling == "perception":
        return {
            "text": "[oracle hint] You are looking for the product shown in the reference image below.",
            "images": [task.anchor_image_path],
        }
    if ceiling == "oracle_retrieval":
        # Synthesise a memory entry for the anchor, bypassing retrieval.
        from dualmem.memory.entry import MemoryEntry
        entry = system.bank.find_by_slug(task.anchor_slug)
        if entry is None:
            cap = system.caption_fn(task.anchor_image_path) if system.caption_fn else None
            entry = MemoryEntry(
                slug=task.anchor_slug,
                image_path=task.anchor_image_path,
                caption=cap,
                encode_text="",
            )
        p = system.oracle_injector.render(entry)
        return {"text": p.text, "images": p.images}
    # full_pipeline
    query = RetrievalQuery(recall_text=task.recall_instruction, anchor_slug=task.anchor_slug)
    retrieved = system.retrieve(query)
    if retrieved is None:
        return {"text": "", "images": []}
    p = system.inject(retrieved)
    return {"text": p.text, "images": p.images}


# ----------------------------------------------------------------------
# Main agent loop
# ----------------------------------------------------------------------

def run_agent_task(
    task: AgentTask,
    system,
    vlm,
    ceiling: str,
    page,
    base_url: str = "http://localhost:3000",
    max_steps: int = 30,
    log_dir: Optional[Path] = None,
) -> AgentRunResult:
    system.reset()
    t0 = time.time()
    transcript_lines: List[str] = [
        f"=== task={task.task_id} system={system.name} ceiling={ceiling} ==="
    ]
    log_dir = log_dir or Path(f"exp/vismem_diag/agent_traces/{task.task_id}_{system.name}_{ceiling}")
    log_dir.mkdir(parents=True, exist_ok=True)

    step = 0

    # Phase 1 — anchor
    page.goto(base_url + task.anchor_url, wait_until="networkidle")
    obs = _capture(page, step, log_dir)
    _encode_observation(system, obs, encode_text=task.recall_instruction)
    transcript_lines.append(f"[step {step}] ANCHOR {obs.url}")
    step += 1

    # Phase 2 — filler
    for url in task.filler_urls:
        page.goto(base_url + url, wait_until="networkidle")
        obs = _capture(page, step, log_dir)
        _encode_observation(system, obs, encode_text="(filler browsing)")
        transcript_lines.append(f"[step {step}] FILLER {obs.url}")
        step += 1

    # Phase 3 — recall loop
    failure = ""
    final_url = obs.url
    noop_streak = 0
    while step < max_steps:
        obs = _capture(page, step, log_dir)
        mem = _memory_payload_for_ceiling(system, task, ceiling)
        prompt = _build_recall_prompt(obs, task, mem.get("text", ""), step, max_steps)
        extra_imgs = mem.get("images", []) or None

        # On a /category page, replace the low-res full-page screenshot with a
        # 2×2 grid of native-resolution per-card crops. Each tile is labelled
        # [0]/[1]/[2]/[3] in DOM order so click_index(N) resolves correctly.
        primary_img = obs.screenshot_path
        grid_path = _capture_collection_grid(page, step, log_dir)
        if grid_path:
            primary_img = grid_path
            transcript_lines.append(
                f"[step {step}] using 2x2 grid screenshot: {grid_path}"
            )

        resp = _vlm_freeform(vlm, system_prompt=SYSTEM_PROMPT, user_text=prompt,
                             primary_image=primary_img, extra_images=extra_imgs)
        transcript_lines.append(f"[step {step}] VLM-raw:")
        for line in resp.split("\n"):
            transcript_lines.append(f"  | {line}")

        action = parse_react_response(resp)
        transcript_lines.append(f"[step {step}] ACTION type={action.type} url={action.url} hash={action.url_hash}")

        if action.type == "add_to_wishlist":
            # If hash specified, navigate first.
            if action.url_hash and obs.product_url_hash != action.url_hash:
                page.goto(base_url + f"/product/{action.url_hash}", wait_until="networkidle")
                obs2 = _capture(page, step + 1, log_dir)
                step += 1
            # Click the wishlist button.
            try:
                page.click("[data-dmv-action='add-to-wishlist']", timeout=2500)
            except Exception as e:
                transcript_lines.append(f"[step {step}] wishlist click failed: {e}")
            final_url = page.url.split(base_url, 1)[-1] or page.url
            transcript_lines.append(f"[step {step}] FINAL after add_to_wishlist: {final_url}")
            break
        if action.type == "done":
            final_url = obs.url
            transcript_lines.append(f"[step {step}] FINAL via done: {final_url}")
            break
        if action.type == "navigate" and action.url:
            try:
                page.goto(base_url + action.url, wait_until="networkidle", timeout=10000)
            except Exception as e:
                transcript_lines.append(f"[step {step}] nav failed: {e}")
            noop_streak = 0
        elif action.type == "click_product" and action.url_hash:
            try:
                page.goto(base_url + f"/product/{action.url_hash}", wait_until="networkidle",
                          timeout=10000)
            except Exception as e:
                transcript_lines.append(f"[step {step}] click_product failed: {e}")
            noop_streak = 0
        elif action.type == "click_index" and action.index is not None:
            # Find the N-th product card on the page and navigate to its href.
            try:
                hrefs = page.eval_on_selector_all(
                    "a[href^='/product/']",
                    "els => Array.from(new Set(els.map(e => e.getAttribute('href'))))"
                )
                idx = action.index
                if 0 <= idx < len(hrefs):
                    page.goto(base_url + hrefs[idx], wait_until="networkidle", timeout=10000)
                    transcript_lines.append(f"[step {step}] click_index({idx}) -> {hrefs[idx]}")
                    noop_streak = 0
                else:
                    transcript_lines.append(f"[step {step}] click_index({idx}) out of range "
                                            f"(only {len(hrefs)} product links on this page)")
                    noop_streak += 1
            except Exception as e:
                transcript_lines.append(f"[step {step}] click_index failed: {e}")
                noop_streak += 1
        else:
            noop_streak += 1
            if noop_streak >= 3:
                failure = "noop_loop"
                final_url = obs.url
                break

        # Encode the post-action page so the system bank reflects what was visited.
        post_obs = _capture(page, step + 1, log_dir)
        _encode_observation(system, post_obs, encode_text="(recall navigation)")
        final_url = post_obs.url
        step += 1
    else:
        failure = "step_cap"

    correct = bool(re.match(task.expected_url_pattern, final_url or ""))
    elapsed = int((time.time() - t0) * 1000)
    transcript_lines.append(f"=== final_url={final_url} correct={correct} steps={step} ===")
    transcript_text = "\n".join(transcript_lines)
    (log_dir / "transcript.txt").write_text(transcript_text)

    return AgentRunResult(
        task_id=task.task_id,
        system=system.name,
        ceiling=ceiling,
        n_steps=step,
        final_url=final_url,
        correct=correct,
        elapsed_ms=elapsed,
        transcript=transcript_text,
        failure_mode=failure,
    )


# ----------------------------------------------------------------------
# VLM free-form text+image call (separate from the 4AFC path in vlm/)
# ----------------------------------------------------------------------

def _vlm_freeform(vlm, system_prompt: str, user_text: str,
                  primary_image: Optional[str] = None,
                  extra_images: Optional[List[str]] = None) -> str:
    """Dispatch a free-form generate call to the active VLM backend.

    All backends in `dualmem.vlm` implement `generate_freeform(...)` —
    Gemini, OpenAI, Anthropic, Qwen-VL, and the stub. Anything without
    that method falls back to the deterministic stub reply so old test
    doubles keep working.
    """
    gen = getattr(vlm, "generate_freeform", None)
    if callable(gen):
        return gen(system_prompt, user_text, primary_image, extra_images, max_tokens=256)
    # Last-resort stub for test doubles that don't implement the protocol.
    return "Thought: stub agent\nAction: navigate(\"/category/chairs?tier=1\")"
