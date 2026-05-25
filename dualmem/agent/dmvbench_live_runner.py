"""DMV-Bench live runner — Memory-Agent-Environment loop for a VISUAL
multimodal agent.

Inspired by MemoryArena's formalism (per-step retrieve + session-end
update) but specialised to **vision-language agents**: at every step the
agent sees not just text but the RENDERED IMAGE of the current page, and
memory entries are multimodal (image + caption + embedding). This is
DMV-Bench's distinguishing axis — MemoryArena tests text-LLM agents on
text-rich environments; we test VLMs on visually-rich shopping pages
with three calibrated grain tiers of perceptual difficulty.

Inner loop (within subtask i):
    For step t = 1..T:
        m_{i,t} = Retrieve(M, s_i, o_{i,1:t-1}, a_{i,1:t-1})        ← per-step
                  visual + text memory; payload may include PNG paths
                  injected as image inputs to the VLM prompt
        a_{i,t} ~ π_A(· | s_i, history, m_{i,t}, o_{i,t}^IMG)        ← real VLM
                  VLM is multimodal: prompt = [system_text, history,
                  current_page_image, retrieved_memory_images,
                  retrieved_caption_text]
        o_{i,t} = Environment(a_{i,t})                                ← JSON GT
Outer loop:
    For subtask i = 1..n:
        run inner loop, accumulate trace
        M ← Update(M, trace)                                          ← session end

Environment is **synthesized from the TaskInstance JSON** — no Playwright,
no dev server. The agent sees the same product images the live website
would serve (from `data/vismem_diag/images/<cat>/var_X_tY.png`) and the
same user messages, but instead of clicking through the website, its
emitted action is compared to GT `expected_url`. This decouples VLM agent
evaluation from Playwright wall-time + dev-server availability.

Per-step the runner saves:
  - log line (Thought/Action/match)
  - PNG screenshot of the page the VLM "saw" this step
  - PNG screenshots of any retrieved memory images injected this step
The screenshot dir mirrors the layout of exp/vismem_diag's Playwright
traces so analysis tooling is shared.

Differences from MemoryArena's text-LLM reference implementation:
  * Multimodal observations and memory injections (image inputs to VLM)
  * Per-tier perceptual difficulty factored separately so visual-acuity
    loss is decoupled from memory-pipeline loss
  * Memory contract per subtask names specific anchor URLs (visual
    items) that must be retrievable; MemoryArena names info units
    abstractly

Cost per task ≈ #agent_turns × 1 Gemini Flash call ≈ $0.025/task @ Flash.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from tasks.schema.multisession import (
    MultiSessionResult,
    MultiSessionTask,
    SessionResult,
    score_cumulative,
)
from tasks.schema.task_instance import TaskInstance, Turn

from dualmem.agent.actions import Action, parse_react_response
from dualmem.agent.multisession_runner import (
    PRODUCT_RE,
    _first_user_cue,
    _load_session_instance,
    _slug_for_url,
    _variant_lookup,
)
from dualmem.retrieval.base import RetrievalQuery


# ---------------------------------------------------------------------------
# Synthetic environment: a turn-indexed simulator of the real website
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """What the agent sees at step t (analogue of a Playwright page snapshot)."""
    url: str
    image_path: str            # disk PNG; "" if no image (category/collection page)
    title: str = ""
    description: str = ""      # the variant's display name + style + category


def _save_placeholder_png(out_path: Path, url: str, title: str) -> None:
    """Render a small text-only PNG for non-product observations so every
    step in a session has a screenshot artifact on disk."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font
    draw.rectangle([8, 8, 504, 248], outline="black", width=2)
    draw.text((20, 30), "[non-product page]", fill="gray", font=font_small)
    draw.text((20, 70), url[:50], fill="black", font=font)
    draw.text((20, 120), title[:60], fill="dimgray", font=font_small)
    draw.text((20, 200), "(no inventory image)", fill="lightgray", font=font_small)
    img.save(out_path)


def _obs_from_url(url: str) -> Observation:
    """Map a /product/<hash> or /category/<slug> URL to a synthetic observation.
    For product URLs, we use the variant_lookup table to fill in image + title.
    """
    var_lookup = _variant_lookup()
    m = PRODUCT_RE.match(url)
    if m:
        entry = var_lookup.get(m.group(1))
        if entry:
            return Observation(
                url=url,
                image_path=entry.get("image_path", ""),
                title=entry.get("text", ""),
                description=entry.get("text", ""),
            )
        return Observation(url=url, image_path="", title=url, description=url)
    return Observation(url=url, image_path="", title=url, description=url)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a shopping-assistant agent in an e-commerce
website. You receive (i) the customer's instruction, (ii) the conversation so
far, (iii) memory context from prior sessions (may be empty), and (iv) the
current page. Decide ONE action.

URL patterns that exist on the site (other patterns return 404):
  /                                  — home
  /category/<slug>                   — category listing, e.g. /category/vase
  /collection/<slug>-<style>         — styled collection, where <style> is
                                        EXACTLY ONE OF: modern, minimalist,
                                        vintage, industrial, scandinavian,
                                        bohemian, mid_century, rustic, japandi,
                                        art_deco
                                        (e.g. /collection/vase-modern,
                                         /collection/chair-rustic)
  /product/<hash>                    — product detail; <hash> is an 8-hex id
  /wishlist /cart                    — shopping pages

Valid category slugs (use EXACTLY these — all singular, no trailing 's'):
  chair, sofa, lamp, cushion, vase, rug, table, bookshelf, plant_pot, wall_art

Action vocabulary (use EXACT syntax — wrong format = no-op):
  navigate("/category/<slug>")           — go to a category listing
  navigate("/collection/<slug>-<style>") — style ∈ {modern, minimalist, vintage, industrial}
  navigate("/product/<hash>")            — go to a product detail page directly
  click_index(N)                         — on a category or collection page,
                                            click the N-th visible product card
                                            (N=0..3, top-left → bottom-right).
                                            PREFERRED over guessing /product/<hash>.
  add_to_wishlist                        — on a product detail page, add it (TERMINAL action,
                                            use it when you've found the right product)
  done                                   — finish if you are stuck (counts as failure)

Some user requests may ask you to navigate to a product the customer has
previously visited. You cannot see earlier sessions in the conversation — only
the [Memory context] block bridges them.

Strategy hints:
  - The [Memory context] block lists products recalled from prior sessions, each
    with its /product/<hash> URL and image. If a remembered product matches what
    the customer is asking for, navigate("/product/<hash>") DIRECTLY — this is
    the fastest and intended path. You DO know the exact hash in that case.
  - Inspect the memory image(s): pick the product whose image matches what the
    customer is describing, then navigate to that product's hash.
  - Only if memory gives you nothing usable: from a category page use
    navigate("/collection/<slug>-<style>") then click_index(N) on the N-th
    product card. You can NOT guess /product/<hash> URLs you have never seen.
  - Once on the /product/<hash> page the customer asked for, emit add_to_wishlist
    — that ends the session successfully.

Reply in ReAct format, ONE Thought line + ONE Action line:
  Thought: <one sentence reasoning>
  Action: <action>
"""


def _format_history(messages: List[dict], max_turns: int = 120) -> str:
    """In-session conversation history — the WHOLE session. Sessions are short
    (a ~10-20-step WebArena-style task), so the agent sees every turn it took
    this session; nothing is windowed away. `max_turns` is only a runaway
    safety cap (keeps the opening task turn + the most recent turns)."""
    if not messages:
        return "(no history yet)"
    if len(messages) > max_turns + 1:
        shown = [messages[0]] + messages[-max_turns:]
        elided = len(messages) - len(shown)
    else:
        shown, elided = messages, 0
    out = []
    for i, m in enumerate(shown):
        prefix = {"user": "User", "agent": "Agent"}.get(m["role"], m["role"])
        out.append(f"  {prefix}: {m['text']}")
        if elided and i == 0:
            out.append(f"  ... ({elided} earlier turns elided) ...")
    return "\n".join(out)


def _format_memory_payload(system, hits) -> Tuple[str, List[str]]:
    """Render up to k retrieved entries through the system's injector, in rank
    order. Returns (text_block, image_paths) where the image_paths line up
    with the 'memory image N' references in the text — so the VLM can pair
    each candidate's URL with its picture.

    Per-baseline modality is preserved: visual injectors attach images, text/
    caption injectors attach none. Every injector now also names the product
    URL (see dualmem.injection.base.entry_url) so the agent can navigate."""
    if not hits:
        return "(no memory retrieved this step)", []
    lines = [
        f"{len(hits)} item(s) recalled from earlier sessions, most relevant "
        f"first. Any attached memory images appear AFTER the current-page "
        f"image, in the order referenced below.",
    ]
    images: List[str] = []
    for i, h in enumerate(hits, 1):
        pl = system.injector.render(h)
        note = ""
        for ip in (pl.images or []):
            if ip:
                images.append(ip)
                note += f" [memory image {len(images)}]"
        body = (pl.text or "(no text)").replace("\n", "\n      ")
        lines.append(f"  ({i}) {body}{note}")
    return "\n".join(lines), images


def _build_prompt(
    subtask_description: str,
    history_messages: List[dict],
    memory_text: str,
    current_obs: Observation,
    step: int,
    cap: int,
) -> str:
    return (
        f"Step {step}/{cap}.\n\n"
        f"[Subtask]\n  {subtask_description}\n\n"
        f"[Conversation so far]\n{_format_history(history_messages)}\n\n"
        f"[Memory context retrieved this step]\n  {memory_text}\n\n"
        f"[Current page]\n  URL: {current_obs.url}\n"
        f"  Title: {current_obs.title}\n"
        f"  Description: {current_obs.description}\n\n"
        "Reply with one Thought + one Action."
    )


# ---------------------------------------------------------------------------
# Action ↔ GT matching (scoring)
# ---------------------------------------------------------------------------

def _action_target_url(action: Action) -> Optional[str]:
    """Pull the URL the agent's action would land on, or None for non-nav."""
    if action.type == "navigate" and action.url:
        return action.url
    if action.type == "click_product" and action.url:
        return action.url
    if action.type == "add_to_wishlist" and action.url:
        return action.url
    return None


def _matches_gt(action: Action, expected_url: Optional[str]) -> bool:
    if not expected_url:
        return action.type in ("done", "add_to_wishlist")
    got = _action_target_url(action)
    return got == expected_url


# ---------------------------------------------------------------------------
# Memory operations — Retrieve / Update wrappers around MemorySystem
# ---------------------------------------------------------------------------

def _retrieve_step(system, query_text: str, k: int = 5):
    """Per-step retrieve. Returns (hits, top-k slugs). The caller renders each
    hit through the system's injector (see _format_memory_payload) so the
    agent gets all k ranked candidates, not just the top-1.

    Results are de-duplicated by slug (keeping the highest-ranked occurrence)
    so the agent never sees the same product twice in one memory block — we
    over-fetch 2k then trim to k distinct slugs."""
    raw = system.retriever.retrieve(
        RetrievalQuery(recall_text=query_text, anchor_slug=None),
        system.bank.entries(), k=k * 2,
    )
    seen: set = set()
    hits = []
    for h in raw:
        if h.slug in seen:
            continue
        seen.add(h.slug)
        hits.append(h)
        if len(hits) >= k:
            break
    return hits, [h.slug for h in hits]


def _bank_has_slug(system, slug: str) -> bool:
    """True if the bank already holds an entry for this slug — a product is
    one memory; re-observing it must not create a duplicate bank entry."""
    return any(getattr(e, "slug", None) == slug for e in system.bank.entries())


def _encode_one_url(system, url: str, caption_fn, log_lines: Optional[list] = None) -> bool:
    """Encode a single visited URL into the bank. Resolves variant lookup
    for product URLs to get product-specific encode_text + image_path.
    Idempotent: a slug already in the bank is skipped (no duplicate entry)."""
    var_lookup = _variant_lookup()
    slug = _slug_for_url(url)
    if _bank_has_slug(system, slug):
        if log_lines is not None:
            log_lines.append(f"    [bank.encode skip-dup] slug={slug}")
        return False
    m = PRODUCT_RE.match(url)
    if m:
        entry = var_lookup.get(m.group(1))
        text = entry["text"] if entry else url
        image_path = entry["image_path"] if entry else ""
    else:
        text = url
        image_path = ""
    cap = caption_fn(image_path) if (caption_fn and image_path) else None
    try:
        system.bank.encode(image_path=image_path, slug=slug,
                            encode_text=text, caption=cap)
        if log_lines is not None:
            log_lines.append(f"    [bank.encode] slug={slug} img={'+' if image_path else '-'} "
                             f"text={text[:60]!r}")
        return True
    except Exception as e:
        if log_lines is not None:
            log_lines.append(f"    [bank.encode fail {slug}: {e}]")
        return False


def _update_session(system, session_trace: List[Tuple[Observation, Action, bool]],
                    caption_fn) -> int:
    """Update(M, trace): write every observed page into the bank at session end.

    For raw banks this is equivalent to incremental encoding; for LLM-summarize
    banks (Mem0-style, not yet implemented) this would call a consolidation
    pass first. Returns # entries added.
    """
    n_added = 0
    for obs, action, ok in session_trace:
        slug = _slug_for_url(obs.url)
        if _bank_has_slug(system, slug):
            continue                       # already a memory — don't duplicate
        text = obs.description
        image_path = obs.image_path
        cap = None
        if caption_fn and image_path:
            cap = caption_fn(image_path)
        system.bank.encode(
            image_path=image_path, slug=slug,
            encode_text=text, caption=cap,
        )
        n_added += 1
    return n_added


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_dmvbench_live(
    task: MultiSessionTask,
    system,
    vlm,
    val_pool: Path,
    log_path: Optional[Path] = None,
    screenshot_dir: Optional[Path] = None,
    max_steps_per_session: int = 20,
    playwright_page = None,
    base_url: str = "http://localhost:3000",
    verbose: bool = True,
) -> MultiSessionResult:
    """Synthetic-env (for VLM input) + optional real Playwright capture.

    log_path           — per (task, system) text transcript (flushed every step)
    screenshot_dir     — directory where per-step PNGs are saved
    playwright_page    — if provided, capture screenshots from the LIVE
                         dev server at base_url for every page (product +
                         category + collection). Disk inventory images are
                         still passed to the VLM as input (deterministic and
                         fast), but the PNG saved to screenshot_dir comes
                         from Playwright for visual ground truth.
    verbose            — stream "step N: action=..." to stdout in real time
    """
    import shutil

    def _flush_log():
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines))

    def _capture_real_screenshot(out_path: Path, url: str) -> bool:
        """If a Playwright page is provided, navigate to base_url+url and
        screenshot. Returns True on success. Used for category/collection
        pages where no static disk image exists.

        We use wait_until='domcontentloaded' (not 'networkidle') because
        Next.js dev maintains small HMR sockets that prevent networkidle
        from firing. domcontentloaded is enough for the static product/
        category page render we want to capture.
        """
        if playwright_page is None:
            return False
        try:
            playwright_page.goto(base_url + url, wait_until="domcontentloaded",
                                  timeout=30000)
            # Wait a short tick so images can stream in before screenshot.
            playwright_page.wait_for_timeout(500)
            playwright_page.screenshot(path=str(out_path), full_page=False)
            return True
        except Exception as e:
            log_lines.append(f"    [playwright capture fail {url}: {e}]")
            return False
    log_lines: List[str] = []
    log_lines.append(f"=== task={task.task_id} variant={task.variant} system={system.name} ===")
    log_lines.append(f"VLM model: {getattr(vlm, 'model', type(vlm).__name__)}")
    log_lines.append(f"cumulative_gt: {task.cumulative_gt}")

    system.reset()
    caption_fn = getattr(system, "caption_fn", None)

    t0 = time.time()
    session_results: List[SessionResult] = []
    cumulative_wishlist: List[str] = []
    total_vlm_calls = 0

    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for i, ref in enumerate(task.sessions):
        inst = _load_session_instance(ref, val_pool)
        s_i = _first_user_cue(inst)
        log_lines.append("")
        log_lines.append(f"--- Session [{i}] {ref.sub_task}  task_id={inst.task_id} ---")
        log_lines.append(f"  s_i (subtask description): {s_i!r}")
        log_lines.append(f"  retrieves_from_prior: {inst.memory_contract.retrieves_from_prior}")
        log_lines.append(f"  must_carry_into_next:   {inst.memory_contract.must_carry_into_next}")
        log_lines.append(f"  bank size at session start: {len(system.bank.entries())}")
        _flush_log()

        # Step-0 cross-session retrieval (for IRR/MRR metrics — diagnostic)
        from dualmem.agent.multisession_runner import _retrieve_for_session
        rmetrics_initial = _retrieve_for_session(system, inst, k=5)
        log_lines.append(f"  initial retrieve (cue=s_i, k=5):")
        log_lines.append(f"    top5: {rmetrics_initial['top_k']}")
        log_lines.append(f"    needs: {inst.memory_contract.retrieves_from_prior}")
        log_lines.append(f"    IRR@5={rmetrics_initial['irr_at_k']} MRR@5={rmetrics_initial['mrr_at_k']} set_hit={rmetrics_initial['set_hit']}")

        # ===== Setup phase: walk ALL non-recall turns so the agent sees the
        # full anchor exposure trace before entering free-form recall.
        #
        # Pre-recall turns include:
        #   - user turns: shown to the VLM as conversation history
        #   - agent turns: Playwright actually navigates to expected_url so
        #     the agent (and the memory pipeline) get to encode that page
        #
        # This handles SA/IC/VL whose 3-5 anchor visits live in agent turns,
        # not user turns. Without this the VLM enters free-form with zero
        # visual exposure to the anchors, which is wrong.
        #
        # Cross-session bridging: if SessionRef carries a preamble (composer-
        # injected for sessions 1..N), prepend it to the FIRST user turn so
        # the prompt explicitly references prior sessions and the memory
        # pipeline is exercised.
        history: List[dict] = []
        preamble = getattr(ref, "cross_session_user_preamble", "") or ""
        first_user_seen = False
        recall_turns_set = set(inst.ncp_metadata.recall_turn_indices)
        first_recall_idx = min(recall_turns_set) if recall_turns_set else 10**9
        setup_navigated_urls: List[str] = []

        # Detect Setup-only session (no agentic free-form needed; we just
        # park the anchor in memory). All XS chains start with one of these.
        is_setup_only = (ref.sub_task == "Setup")

        for turn in inst.turns:
            # Stop once we hit the first recall turn — free-form takes over.
            # Exception: Setup sessions, where the "recall" is the anchor
            # navigation itself and we want to execute it directly.
            if turn.turn_index >= first_recall_idx and not is_setup_only:
                break
            if turn.role == "system":
                continue
            if turn.role == "user" and turn.content:
                text = turn.content
                if not first_user_seen and preamble:
                    text = preamble + text
                first_user_seen = True
                history.append({"role": "user", "text": text})
                continue
            if turn.role == "agent" and turn.expected_url:
                # SPEEDUP (lever 1): the setup-phase browse does NOT need a
                # real browser navigation. _encode_one_url reads the product
                # image straight from disk, and setup navs never produce a
                # screenshot. Skipping the Playwright goto removes ~100 page
                # loads per task — the dominant per-task cost. We still record
                # the visit in history and encode it into the bank, so the
                # agent's history and the memory pipeline are unchanged.
                setup_navigated_urls.append(turn.expected_url)
                history.append({
                    "role": "agent",
                    "text": f'(browsed) {turn.expected_url}',
                })
                _encode_one_url(system, turn.expected_url,
                                caption_fn=caption_fn, log_lines=log_lines)

        if preamble:
            log_lines.append(f"  cross-session preamble applied: {preamble!r}")
        log_lines.append(f"  setup phase: {len(history)} turns into VLM history; "
                         f"navigated {len(setup_navigated_urls)} anchor URLs")
        for u in setup_navigated_urls:
            log_lines.append(f"    [setup nav] {u}")
        gt_url = inst.ground_truth.target_url
        log_lines.append(f"  ground_truth.target_url: {gt_url}")
        _flush_log()

        # Setup phase no longer drives the browser (lever 1) — park Playwright
        # at home so the free-form phase starts from a clean, known page.
        if playwright_page is not None and not is_setup_only:
            try:
                playwright_page.goto(base_url + "/", wait_until="domcontentloaded",
                                     timeout=30000)
            except Exception as e:
                log_lines.append(f"  [setup goto / fail: {e}]")

        # Setup sessions don't need free-form: the anchor was just navigated
        # to + encoded above. Trivially mark correct and continue.
        if is_setup_only:
            log_lines.append(f"  [Setup session]: skipping free-form. "
                             f"anchor encoded, session_correct=True")
            session_results.append(SessionResult(
                task_id=inst.task_id, sub_task=inst.sub_task, order_index=i,
                correct=True, final_url=setup_navigated_urls[-1] if setup_navigated_urls else "/",
                final_wishlist=list(cumulative_wishlist),
                n_steps=0,
                retrieval_top5=None, irr_at_5=None, mrr_at_5=None,
                retrieval_set_hit=None, failure_mode="",
            ))
            for vid in inst.memory_contract.must_carry_into_next[:1]:
                if vid not in cumulative_wishlist:
                    cumulative_wishlist.append(vid)
            _flush_log()
            continue

        # ===== Free-form phase =====
        # Agent decides actions; Playwright executes; loop until terminal or
        # max_steps_per_session reached. last_product_url starts unset — only
        # free-form navigations count toward "the agent's final answer".
        trace: List[Tuple[Observation, Action, bool]] = []
        any_action_matched = False
        last_product_url: Optional[str] = None
        terminated_by = "step_cap"

        for step_count in range(1, max_steps_per_session + 1):
            # Capture current page from live Playwright session
            current_url = playwright_page.url if playwright_page else "/"
            rel_url = current_url
            if base_url in rel_url:
                rel_url = rel_url.split(base_url, 1)[-1] or "/"
            current_obs = _obs_from_url(rel_url)
            # Prefer Playwright live screenshot for the obs image
            obs_screenshot_path = None
            if screenshot_dir is not None:
                global_step += 1
                step_prefix = f"sess{i:02d}_step{step_count:02d}_g{global_step:03d}"
                obs_out = screenshot_dir / f"{step_prefix}_obs.png"
                saved = False
                if playwright_page is not None:
                    try:
                        playwright_page.wait_for_timeout(200)
                        playwright_page.screenshot(path=str(obs_out), full_page=False)
                        saved = True
                        obs_screenshot_path = str(obs_out)
                    except Exception as e:
                        log_lines.append(f"    [playwright screenshot fail: {e}]")
                if not saved and current_obs.image_path:
                    try:
                        shutil.copy(current_obs.image_path, obs_out)
                        saved = True
                        obs_screenshot_path = str(obs_out)
                    except Exception as e:
                        log_lines.append(f"    [screenshot copy fail: {e}]")
                if not saved:
                    try:
                        _save_placeholder_png(obs_out, current_obs.url, current_obs.title)
                        obs_screenshot_path = str(obs_out)
                    except Exception as e:
                        log_lines.append(f"    [screenshot placeholder fail: {e}]")

            # Per-step retrieve.
            query_text = f"{s_i} CURRENT: {current_obs.url}"
            hits, top_k = _retrieve_step(system, query_text, k=5)
            mem_text, mem_images = _format_memory_payload(system, hits)

            prompt = _build_prompt(
                subtask_description=s_i,
                history_messages=history,
                memory_text=mem_text,
                current_obs=current_obs,
                step=step_count,
                cap=max_steps_per_session,
            )
            primary = obs_screenshot_path or current_obs.image_path or None
            extras = mem_images or None
            try:
                a_text = vlm.generate_freeform(
                    system_prompt=SYSTEM_PROMPT_TEMPLATE,
                    user_text=prompt,
                    primary_image=primary,
                    extra_images=extras,
                    max_tokens=512,
                )
            except Exception as e:
                a_text = f"Thought: vlm error\nAction: done   # err: {type(e).__name__}: {e}"
            total_vlm_calls += 1

            action = parse_react_response(a_text)

            log_lines.append(f"  STEP {step_count}/{max_steps_per_session}")
            log_lines.append(f"    obs.url={current_obs.url}")
            log_lines.append(f"    retrieved.top5 = {top_k}")
            log_lines.append(f"    memory injected to VLM ({len(mem_images)} image(s)):")
            for line in mem_text.splitlines():
                log_lines.append(f"      {line}")
            for line in a_text.strip().splitlines():
                log_lines.append(f"    | {line}")
            log_lines.append(f"    parsed: type={action.type} url={action.url} "
                             f"hash={action.url_hash} index={action.index}")

            # Save memory-image screenshots (only the *_obs.png already saved above)
            if screenshot_dir is not None:
                for k_idx, ip in enumerate(mem_images or []):
                    try:
                        shutil.copy(ip, screenshot_dir / f"{step_prefix}_mem{k_idx}.png")
                    except Exception as e:
                        log_lines.append(f"    [mem screenshot fail {k_idx}: {e}]")

            # Execute action against Playwright
            terminal = False
            executed_target_url: Optional[str] = None
            if playwright_page is None:
                log_lines.append(f"    [no playwright; action noop]")
            else:
                try:
                    if action.type == "navigate" and action.url:
                        playwright_page.goto(base_url + action.url,
                                             wait_until="domcontentloaded", timeout=30000)
                        executed_target_url = action.url
                    elif action.type == "click_product" and action.url:
                        playwright_page.goto(base_url + action.url,
                                             wait_until="domcontentloaded", timeout=30000)
                        executed_target_url = action.url
                    elif action.type == "click_index" and action.index is not None:
                        hrefs = playwright_page.eval_on_selector_all(
                            "a[href^='/product/']",
                            "els => Array.from(new Set(els.map(e => e.getAttribute('href'))))",
                        )
                        idx = action.index
                        if 0 <= idx < len(hrefs):
                            playwright_page.goto(base_url + hrefs[idx],
                                                 wait_until="domcontentloaded", timeout=30000)
                            executed_target_url = hrefs[idx]
                            log_lines.append(f"    click_index({idx}) → {hrefs[idx]}")
                        else:
                            log_lines.append(f"    click_index({idx}) out of range "
                                             f"(only {len(hrefs)} product links)")
                    elif action.type == "add_to_wishlist":
                        # Optionally navigate first if a hash was emitted
                        if action.url_hash:
                            target = f"/product/{action.url_hash}"
                            if target not in playwright_page.url:
                                playwright_page.goto(base_url + target,
                                                     wait_until="domcontentloaded",
                                                     timeout=30000)
                                executed_target_url = target
                        try:
                            playwright_page.click("[data-dmv-action='add-to-wishlist']",
                                                  timeout=2500)
                        except Exception as e:
                            log_lines.append(f"    wishlist click fail: {e}")
                        terminal = True
                        terminated_by = "add_to_wishlist"
                    elif action.type == "done":
                        terminal = True
                        terminated_by = "done"
                    else:
                        log_lines.append(f"    [noop action: {action.type}]")
                except Exception as e:
                    log_lines.append(f"    [exec error: {e}]")

            # Track last product page the agent landed on
            final_rel = playwright_page.url if playwright_page else current_obs.url
            if base_url in final_rel:
                final_rel = final_rel.split(base_url, 1)[-1] or "/"
            if PRODUCT_RE.match(final_rel.split("?")[0]):
                last_product_url = final_rel.split("?")[0]
                log_lines.append(f"    last_product_url ← {last_product_url}")

            # Append for trace + history
            trace.append((current_obs, action, True))
            history.append({"role": "agent", "text": a_text})

            if verbose:
                short_obs = (current_obs.url or "")[:30]
                print(f"      step {step_count:2d} sess{i} obs={short_obs:30s} "
                      f"action={action.type:10s} → {executed_target_url or 'no-nav'}",
                      flush=True)
            _flush_log()

            if terminal:
                log_lines.append(f"  [terminal: {terminated_by} @ step {step_count}]")
                break
        else:
            log_lines.append(f"  [step cap {max_steps_per_session} reached]")

        # End-of-session: Update(M, trace)
        n_added = _update_session(system, trace, caption_fn)
        log_lines.append(f"  Update(M, trace): added {n_added} entries to bank")
        log_lines.append(f"  bank size at session end: {len(system.bank.entries())}")

        # Score: did the agent's final landing product match GT?
        session_correct = (last_product_url is not None and gt_url is not None
                           and last_product_url == gt_url)
        log_lines.append(f"  last_product_url: {last_product_url}")
        log_lines.append(f"  gt_url:           {gt_url}")
        log_lines.append(f"  session_correct:  {session_correct}")
        final_url = last_product_url or "/"

        # Cumulative wishlist tracking (for long-horizon variants).
        if session_correct:
            for vid in inst.memory_contract.must_carry_into_next[:1]:
                if vid not in cumulative_wishlist:
                    cumulative_wishlist.append(vid)
            if (inst.ground_truth.target_variant_id
                    and inst.ground_truth.target_variant_id not in cumulative_wishlist):
                cumulative_wishlist.append(inst.ground_truth.target_variant_id)

        session_results.append(SessionResult(
            task_id=inst.task_id,
            sub_task=inst.sub_task,
            order_index=i,
            correct=session_correct,
            final_url=final_url,
            final_wishlist=list(cumulative_wishlist),
            n_steps=step_count,
            retrieval_top5=rmetrics_initial["top_k"] or None,
            irr_at_5=rmetrics_initial["irr_at_k"],
            mrr_at_5=rmetrics_initial["mrr_at_k"],
            retrieval_set_hit=rmetrics_initial["set_hit"],
            failure_mode="" if session_correct else "vlm_wrong_recall",
        ))

    cumulative_ok = score_cumulative(task, cumulative_wishlist)
    all_ok = all(s.correct for s in session_results)

    # cumulative_correct=True when cumulative_gt is empty is trivially true
    # (the variant didn't declare a cross-session GT constraint). Display
    # "N/A" so it's not confused with "passed".
    cum_display = "N/A (no cumulative GT)" if not task.cumulative_gt else cumulative_ok

    log_lines.append("")
    log_lines.append("=== FINAL ===")
    log_lines.append(f"  cumulative_gt:       {task.cumulative_gt}")
    log_lines.append(f"  cumulative_wishlist: {cumulative_wishlist}")
    log_lines.append(f"  cumulative_correct:  {cum_display}")
    log_lines.append(f"  per_session_correct: "
                     f"{['Pass' if s.sub_task == 'Setup' else s.correct for s in session_results]}")
    log_lines.append(f"  all_sessions_ok:     {all_ok}")
    log_lines.append(f"  total Gemini calls:  {total_vlm_calls}")
    log_lines.append(f"  elapsed_ms:          {int((time.time()-t0)*1000)}")
    log_lines.append(f"  task_correct:        {all_ok and cumulative_ok}  "
                     f"(= all_sessions AND cumulative_constraint_met)")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines))

    return MultiSessionResult(
        task_id=task.task_id,
        variant=task.variant,
        sessions=session_results,
        correct=(all_ok and cumulative_ok),
        cumulative_correct=cumulative_ok,
        elapsed_ms=int((time.time() - t0) * 1000),
        notes=f"live_runner, vlm_calls={total_vlm_calls}",
    )
