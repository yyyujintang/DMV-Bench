"""MultiSessionRunner — walks a MultiSessionTask, drives the agent through
each session in turn with VLM context resets at session boundaries.

One run produces a MultiSessionResult with:
  - per-session correctness (URL-match or variant-match)
  - cross-session retrieval correctness (was the right anchor on top-1 at
    each session boundary?)
  - cumulative GT correctness (final wishlist contains all anchors from
    earlier sessions, for long-horizon variants)

The runner has two modes:
  - `mode='live'`: drives a Playwright `page` through every turn's
    `expected_url`. Used for real evals.
  - `mode='oracle'`: skip Playwright entirely; replay the TaskInstance's
    declared visit sequence as if the agent had executed it perfectly.
    Used to (a) test the memory pipeline in isolation, (b) sanity-check
    multi-session composition while the website is being refactored.

In both modes:
  - Memory bank persists across sessions in a task.
  - VLM context is fresh per session (the runner builds the prompt anew).
  - Memory bank resets at the start of every TASK (multisession_design.md §F.4).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional

from tasks.schema.multisession import (
    MultiSessionResult,
    MultiSessionTask,
    SessionResult,
    score_cumulative,
    score_session,
)
from tasks.schema.task_instance import TaskInstance

from dualmem.agent.observation import PageObservation
from dualmem.retrieval.base import RetrievalQuery


# Map a TaskInstance turn's `expected_url` (e.g. "/product/abc12345") to a
# stable slug — for product pages we use the hash, for category pages we
# use the slug-tier combo. The slug is the key the memory bank stores by.
PRODUCT_RE = re.compile(r"^/product/([0-9a-f]{6,16})$")
CATEGORY_RE = re.compile(r"^/category/([a-z_-]+)")
COLLECTION_RE = re.compile(r"^/collection/([a-z_-]+)")


def _slug_for_url(url: str) -> str:
    if not url:
        return "blank"
    m = PRODUCT_RE.match(url)
    if m:
        return m.group(1)
    m = COLLECTION_RE.match(url)
    if m:
        return f"col_{m.group(1)}"
    m = CATEGORY_RE.match(url)
    if m:
        return f"cat_{m.group(1)}"
    return f"page_{abs(hash(url)) % 100000}"


# Lazy-loaded variant catalogue → urlHash → (encode_text, image_abs_path).
# encode_text breaks SBERT-tie issues (each variant gets product-specific text).
# image_abs_path lets visual encoders (CoMEM/HYMEM/DualChannel) actually run
# CLIP/DINOv2/DINOv3 on the real product image instead of seeing image_path="".
_VARIANT_LOOKUP_CACHE: Optional[dict] = None
_DEFAULT_IMAGE_ROOT: Optional[Path] = None


def _variant_lookup() -> dict:
    """Map urlHash → {'text': str, 'image_path': str} for the v2 inventory.

    Loads `data/vismem_diag_v2/cue_registry.json` (1000 rows, one per
    product, each with url_hash + cat/style/prod_idx). image_path resolves
    to the with_cue PNG under
        data/vismem_diag_v2/images/with_cue/<cat>/<style>/<prod_idx>.png

    encode_text is the BARE category noun ("a vase") — never the cue,
    style, or color. The incidental cue must be recalled VISUALLY through
    the image; if encode_text named the cue, text-only baselines would
    cheat the IC benchmark.
    """
    global _VARIANT_LOOKUP_CACHE, _DEFAULT_IMAGE_ROOT
    if _VARIANT_LOOKUP_CACHE is None:
        try:
            import json as _json
            v2_root = (_PROJECT_ROOT_FOR_V2()
                       / "data" / "vismem_diag_v2")
            registry = _json.loads((v2_root / "cue_registry.json").read_text())
            with_cue_root = v2_root / "images" / "with_cue"
            _DEFAULT_IMAGE_ROOT = with_cue_root
            out: dict[str, dict] = {}
            for r in registry["rows"]:
                cat, style, idx = r["cat"], r["style"], r["prod_idx"]
                img = with_cue_root / cat / style / f"{idx:02d}.png"
                out[r["url_hash"]] = {
                    "text": f"a {cat.replace('_', ' ')}",
                    "image_path": str(img) if img.exists() else "",
                }
            _VARIANT_LOOKUP_CACHE = out
        except Exception:
            _VARIANT_LOOKUP_CACHE = {}
    return _VARIANT_LOOKUP_CACHE


def _PROJECT_ROOT_FOR_V2() -> Path:
    """Repository root — two parents up from this file (dualmem/agent/…)."""
    return Path(__file__).resolve().parents[2]


# Backwards-compat shim — earlier code path used a flat dict of urlHash → text.
def _variant_text_lookup() -> dict:
    return {h: d["text"] for h, d in _variant_lookup().items()}


def _load_session_instance(session_ref, val_pool: Path) -> TaskInstance:
    """Load a sub-task TaskInstance JSON from `tasks/pool/validated/<SUB>/`,
    applying any retrieves_from_prior_override / must_carry_override on the
    SessionRef so the runner sees the composer's cross-session wiring.

    PD-long variants live in the PD subdir with a `pd_long_` filename prefix.
    """
    sub_dir = "PD" if session_ref.task_id.startswith("pd_long_") else session_ref.sub_task
    path = val_pool / sub_dir / f"{session_ref.task_id}.json"
    inst = TaskInstance.model_validate_json(path.read_text())
    if session_ref.retrieves_from_prior_override or session_ref.must_carry_override:
        new_contract = inst.memory_contract.model_copy(update={
            "retrieves_from_prior": list(session_ref.retrieves_from_prior_override)
              or list(inst.memory_contract.retrieves_from_prior),
            "must_carry_into_next": list(session_ref.must_carry_override)
              or list(inst.memory_contract.must_carry_into_next),
        })
        inst = inst.model_copy(update={"memory_contract": new_contract})
    return inst


def _encode_turn_pages(
    system,
    turns,
    encode_text_default: str,
    image_root: Optional[Path] = None,
    log_lines: Optional[list] = None,
) -> None:
    """Oracle-replay encoding step (fix A + E from the design discussion).

    Two encoding sources per session:

    A. **User-referenced variants** — when a user turn carries
       `references_variant` (e.g. NC's "Here's one I like — show me cheapest
       similar"), oracle treats it as if the user had visually shown the
       anchor to the agent, so we encode it. Without this, NC's anchor never
       lands in the bank and cross-session NC retrieval is trivially False.

    B. **Agent-visited URLs** — every agent turn with an `expected_url`.
       Each entry's `encode_text` is the variant's product-specific
       descriptor (`Modern Accent Chair — Caramel (modern, chairs)`) rather
       than a uniform per-session string. This breaks SBERT ties so the
       retriever scores entries on actual semantic similarity to the cue
       instead of degenerating to insertion order.
    """
    var_lookup = _variant_lookup()
    caption_fn = getattr(system, "caption_fn", None)
    # In oracle mode, the "caption" the agent would have seen is approximated
    # by the variant's display_name (since we're not running a real VLM). We
    # therefore prefer the variant_lookup text as caption directly, regardless
    # of whether a disk image exists — many categories (bookshelves, plant_pots,
    # sofas, wall_art) don't have inventory PNGs but we still want
    # Caption/DualChannel/HYMEM to see a real text representation.
    def _caption_for(vid: Optional[str], img_path: str) -> Optional[str]:
        if not caption_fn:
            return None
        entry = var_lookup.get(vid) if vid else None
        if entry and entry.get("text"):
            return entry["text"]
        if img_path:
            return caption_fn(img_path)
        return None
    for turn in turns:
        # A — user-referenced variant ("I like this chair I'm pointing at")
        if turn.role == "user" and turn.references_variant:
            vid = turn.references_variant
            entry = var_lookup.get(vid)
            text = entry["text"] if entry else (turn.content or encode_text_default)
            img = entry["image_path"] if entry else ""
            cap = _caption_for(vid, img)
            system.bank.encode(
                image_path=img, slug=vid, encode_text=text, caption=cap,
            )
            if log_lines is not None:
                log_lines.append(
                    f"    ENCODE(user_ref) slug={vid:20s} img={'+' if img else '-'} "
                    f"text={text[:60]!r} caption={(cap or '')[:60]!r}"
                )
        # B — agent-visited page
        if not turn.expected_url:
            continue
        slug = _slug_for_url(turn.expected_url)
        # E — product-specific encode_text + image where possible
        m = PRODUCT_RE.match(turn.expected_url)
        if m:
            entry = var_lookup.get(m.group(1))
            text = entry["text"] if entry else encode_text_default
            image_path = entry["image_path"] if entry else ""
        else:
            text = turn.expected_url   # collection / category URL itself is descriptive
            image_path = ""
        # Caller-supplied image_root overrides; useful for synthetic tests.
        if image_root is not None and not image_path:
            candidate = image_root / f"{slug}.png"
            if candidate.exists():
                image_path = str(candidate)
        cap = caption_fn(image_path) if (caption_fn and image_path) else None
        system.bank.encode(
            image_path=image_path,
            slug=slug,
            encode_text=text,
            caption=cap,
        )
        if log_lines is not None:
            log_lines.append(
                f"    ENCODE slug={slug:24s}  img={'+' if image_path else '-'}  "
                f"text={text[:60]!r}  caption={cap[:60]!r}" if cap else
                f"    ENCODE slug={slug:24s}  img={'+' if image_path else '-'}  "
                f"text={text[:60]!r}  caption=None"
            )


def _retrieve_for_session_oracle_bypass(
    system,
    session_instance: TaskInstance,
    k: int = 5,
) -> dict:
    """Oracle-retrieval ceiling: skip the real retriever, look up needs by slug
    in the bank. If found, put them at top of `top_k`; pad with any other entries.
    Tells us "if retrieval was perfect, IRR@5/MRR@5 = 1.0".
    """
    needs = list(session_instance.memory_contract.retrieves_from_prior)
    if not needs:
        return {"top_k": [], "irr_at_k": None, "mrr_at_k": None, "set_hit": None}
    # Pull every entry from bank that matches a need; preserve need-order.
    all_entries = system.bank.entries()
    by_slug = {e.slug: e for e in all_entries}
    hits = [by_slug[n] for n in needs if n in by_slug]
    pad = [e for e in all_entries if e not in hits][: max(0, k - len(hits))]
    top_k_entries = (hits + pad)[:k]
    top_k = [e.slug for e in top_k_entries]
    needs_set = set(needs)
    intersection = [s for s in top_k if s in needs_set]
    irr = len(set(intersection)) / len(needs_set) if needs_set else None
    mrr = 0.0
    for rank, slug in enumerate(top_k, start=1):
        if slug in needs_set:
            mrr = 1.0 / rank
            break
    return {
        "top_k": top_k,
        "irr_at_k": irr,
        "mrr_at_k": mrr,
        "set_hit": bool(intersection),
    }


def _retrieve_for_session(
    system,
    session_instance: TaskInstance,
    k: int = 5,
) -> dict:
    """Query memory at the start of a session that declares
    `retrieves_from_prior`. Returns the metrics dict per §G.2 of the
    multi-session design:

        {
            "top_k": [slug, ...],        # length ≤ k, ordered by retriever rank
            "irr_at_k": float | None,    # |top_k ∩ needs| / |needs|
            "mrr_at_k": float | None,    # 1 / (1-indexed rank of first hit); 0 if none
            "set_hit": bool | None,      # any top_k ∈ needs
        }

    When `needs` is empty (session 0 or no cross-session contract), all
    metrics are None — caller should skip them in aggregation.
    """
    needs = list(session_instance.memory_contract.retrieves_from_prior)
    if not needs:
        return {"top_k": [], "irr_at_k": None, "mrr_at_k": None, "set_hit": None}
    # The query is the recall instruction — NOT the browse-intent turn that
    # opens a relay session. _first_user_cue prefers the recall-mode user turn.
    cue = _first_user_cue(session_instance)
    query = RetrievalQuery(recall_text=cue, anchor_slug=None)
    hits = system.retriever.retrieve(query, system.bank.entries(), k=k)
    top_k = [h.slug for h in hits]
    needs_set = set(needs)
    intersection = [s for s in top_k if s in needs_set]
    irr = len(set(intersection)) / len(needs_set)
    mrr = 0.0
    for rank, slug in enumerate(top_k, start=1):
        if slug in needs_set:
            mrr = 1.0 / rank
            break
    return {
        "top_k": top_k,
        "irr_at_k": irr,
        "mrr_at_k": mrr,
        "set_hit": bool(intersection),
    }


def _first_user_cue(session_instance: TaskInstance) -> str:
    """The session's actionable cue — the user instruction the agent must
    satisfy. In a relay session the first user turn is only a browse-intent
    ("let me look at some lamps"); the real task is the recall request, a
    user turn with mode=='recall'. Prefer that; fall back to the last user
    turn, then the first (preserves single-user-turn sessions)."""
    recall_user = [t.content for t in session_instance.turns
                   if t.role == "user" and t.content and t.mode == "recall"]
    if recall_user:
        return recall_user[-1]
    user_turns = [t.content for t in session_instance.turns
                  if t.role == "user" and t.content]
    return user_turns[-1] if user_turns else ""


def _final_url_of_session(session_instance: TaskInstance) -> str:
    """Oracle mode: the agent does exactly what the GT prescribes. The 'final'
    URL is the last agent turn's expected_url (or ground_truth.target_url
    as fallback)."""
    last_url = ""
    for t in reversed(session_instance.turns):
        if t.role == "agent" and t.expected_url:
            last_url = t.expected_url
            break
    return last_url or (session_instance.ground_truth.target_url or "")


def run_multi_session_oracle(
    task: MultiSessionTask,
    system,
    val_pool: Path,
    image_root: Optional[Path] = None,
    log_path: Optional[Path] = None,
    ceiling: str = "full_pipeline",
) -> MultiSessionResult:
    """Oracle-agent replay across one of three memory ceilings.

    Ceilings (multisession analogue of VisMem-Diag §3.3):
      - "perception"       : bank reset between sessions → no cross-session
                             memory possible. Tells us SR floor when memory
                             pipeline contributes nothing.
      - "oracle_retrieval" : retrieve magically returns an entry matching
                             retrieves_from_prior (if the bank holds one).
                             Tells us SR when retrieval is perfect; gap to
                             full_pipeline = retrieval loss.
      - "full_pipeline"    : everything real (default).
    """
    assert ceiling in ("perception", "oracle_retrieval", "full_pipeline")
    """Oracle-replay runner. The agent is assumed to execute the GT actions
    in every session. The runner exercises the memory pipeline only —
    encoding observations and retrieving across session boundaries.

    Used to:
      - validate that memory_contract wiring is right
      - measure the memory-pipeline's retrieval@1 accuracy at session
        boundaries given perfect agent execution
      - bootstrap the cross-session gap heatmap while the live website
        refactor settles
    """
    t0 = time.time()
    system.reset()
    session_results: List[SessionResult] = []
    cumulative_wishlist: List[str] = []
    log_lines: List[str] = []
    log_lines.append(f"=== task={task.task_id}  variant={task.variant}  system={system.name} ===")
    log_lines.append(f"cumulative_gt: {task.cumulative_gt}")

    for i, ref in enumerate(task.sessions):
        # Perception ceiling: bank is wiped at every session boundary →
        # cross-session memory cannot work at all.
        if ceiling == "perception" and i > 0:
            system.bank.reset()
            log_lines.append("[perception ceiling] bank reset before session start")
        inst = _load_session_instance(ref, val_pool)
        log_lines.append("")
        log_lines.append(f"--- session [{i}] {ref.sub_task} task_id={inst.task_id} ---")
        log_lines.append(f"  memory_contract.encodes              = {inst.memory_contract.encodes}")
        log_lines.append(f"  memory_contract.retrieves_from_prior = {inst.memory_contract.retrieves_from_prior}")
        log_lines.append(f"  memory_contract.must_carry_into_next = {inst.memory_contract.must_carry_into_next}")
        log_lines.append(f"  bank size before session: {len(system.bank.entries())}")

        # Cross-session retrieval check at session boundary (top-k metrics).
        if ceiling == "oracle_retrieval":
            rmetrics = _retrieve_for_session_oracle_bypass(system, inst, k=5)
            log_lines.append("  [oracle_retrieval ceiling] retriever bypassed; slug-lookup")
        else:
            rmetrics = _retrieve_for_session(system, inst, k=5)
        log_lines.append(f"  RETRIEVE cue: {_first_user_cue(inst)[:80]!r}")
        log_lines.append(f"  RETRIEVE top-5: {rmetrics['top_k']}")
        log_lines.append(f"  RETRIEVE needs: {inst.memory_contract.retrieves_from_prior}")
        log_lines.append(f"  RETRIEVE → IRR@5={rmetrics['irr_at_k']} MRR@5={rmetrics['mrr_at_k']} set_hit={rmetrics['set_hit']}")

        # Encode every page the GT visits in this session.
        log_lines.append(f"  ENCODE pass (oracle replay of GT actions):")
        _encode_turn_pages(
            system, inst.turns,
            encode_text_default=f"session {i}: {ref.sub_task}",
            image_root=image_root,
            log_lines=log_lines,
        )
        log_lines.append(f"  bank size after session : {len(system.bank.entries())}")

        # "Final URL" of session = the last agent turn (oracle execution).
        final_url = _final_url_of_session(inst)
        # In oracle replay, score_session(final_url == GT) is trivially True by
        # construction (the agent reads GT from JSON). To make SR informative,
        # we redefine session_correct in oracle mode as:
        #     session 0 (no retrieval needed) → trivially True
        #     session N > 0                   → retrieval_set_hit (top-5 ∩ needs ≠ ∅)
        # Real task-success scoring (URL match etc.) is unchanged — `correct_url`
        # below is retained for diagnostics but does NOT drive SR/PS in oracle.
        correct_url = score_session(inst, final_url, final_wishlist=cumulative_wishlist)
        if rmetrics["set_hit"] is None:
            # No retrieval contract — session 0 or attention-check
            correct = correct_url
        else:
            correct = bool(rmetrics["set_hit"])
        if correct_url:
            # In live mode the agent toggles a wishlist button per session.
            # In oracle, simulate: add the session's "to-carry" anchor (first
            # must_carry_into_next id) PLUS its own target_variant_id. The
            # cumulative_gt compiled by the composer is the union of those
            # per-session anchors + the final session's target, so this keeps
            # the two in sync.
            for vid in inst.memory_contract.must_carry_into_next[:1]:
                if vid not in cumulative_wishlist:
                    cumulative_wishlist.append(vid)
            if inst.ground_truth.target_variant_id and inst.ground_truth.target_variant_id not in cumulative_wishlist:
                cumulative_wishlist.append(inst.ground_truth.target_variant_id)
        session_results.append(SessionResult(
            task_id=inst.task_id,
            sub_task=inst.sub_task,
            order_index=i,
            correct=correct,
            final_url=final_url,
            final_wishlist=list(cumulative_wishlist),
            n_steps=len(inst.turns),
            retrieval_top5=rmetrics["top_k"] or None,
            irr_at_5=rmetrics["irr_at_k"],
            mrr_at_5=rmetrics["mrr_at_k"],
            retrieval_set_hit=rmetrics["set_hit"],
            failure_mode="",
        ))

    cumulative_ok = score_cumulative(task, cumulative_wishlist)
    all_sessions_ok = all(s.correct for s in session_results)
    log_lines.append("")
    log_lines.append("=== FINAL ===")
    log_lines.append(f"  cumulative_wishlist: {cumulative_wishlist}")
    log_lines.append(f"  cumulative_correct : {cumulative_ok}")
    log_lines.append(f"  all_sessions_ok   : {all_sessions_ok}")
    log_lines.append(f"  per_session correct      : {[s.correct for s in session_results]}")
    log_lines.append(f"  per_session set_hit (oracle SR): {[s.retrieval_set_hit for s in session_results]}")
    log_lines.append(f"  per_session IRR@5: {[s.irr_at_5 for s in session_results]}")
    log_lines.append(f"  per_session MRR@5: {[s.mrr_at_5 for s in session_results]}")
    log_lines.append(f"  task_correct: {all_sessions_ok and cumulative_ok}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines))
    return MultiSessionResult(
        task_id=task.task_id,
        variant=task.variant,
        sessions=session_results,
        correct=(all_sessions_ok and cumulative_ok),
        cumulative_correct=cumulative_ok,
        elapsed_ms=int((time.time() - t0) * 1000),
        notes=f"oracle replay, n_sessions={len(task.sessions)}",
    )


def run_multi_session_live(
    task: MultiSessionTask,
    system,
    vlm,
    page,                      # Playwright Page (live website)
    val_pool: Path,
    base_url: str = "http://localhost:3000",
    log_root: Optional[Path] = None,
) -> MultiSessionResult:
    """Live runner — drives Playwright through each session's expected
    actions, but at every recall turn calls the VLM to decide the next
    navigation/click. The agent's final action per session decides per-
    session TSR. Memory bank persists across sessions within a task.

    This wraps the existing single-session `run_agent_task` for each
    sub-task in the multi-session bundle. Between sub-tasks, a fresh
    VLM context is implicitly built by run_agent_task (it constructs its
    own prompts from scratch each call).
    """
    from dualmem.agent.playwright_agent import run_agent_task
    from dualmem.tasks.spec import AgentTask

    t0 = time.time()
    system.reset()
    session_results: List[SessionResult] = []
    cumulative_wishlist: List[str] = []
    log_root = log_root or Path(f"exp/vismem_diag/multisession/{task.task_id}")

    for i, ref in enumerate(task.sessions):
        inst = _load_session_instance(ref, val_pool)
        # Adapt the TaskInstance into the AgentTask 3-phase shape so we can
        # reuse the existing single-session agent loop. We extract:
        #   anchor_url       = first agent turn with expected_url on /product/*
        #   filler_urls      = subsequent agent turn URLs before recall
        #   recall_collection_url = the recall turn's expected_url (if any)
        #                            or the last category URL seen
        #   expected_url_pattern  = regex for ground_truth.target_url
        anchor_url, filler_urls, recall_url = _adapt_turns_to_phases(inst)
        adapt = AgentTask(
            task_id=inst.task_id,
            mechanism=inst.sub_task.lower(),
            category=inst.category_ids[0] if inst.category_ids else "unknown",
            grain_tier=inst.grain_tier,
            leakage_level=4,
            seed=0,
            anchor_url=anchor_url,
            anchor_slug=(inst.ncp_metadata.anchor_variant_ids[:1] or ["anchor"])[0],
            anchor_image_path="",
            filler_urls=filler_urls,
            recall_instruction=_recall_instruction(inst),
            recall_collection_url=recall_url,
            expected_url_pattern=f"^{re.escape(inst.ground_truth.target_url or '')}",
            candidate_urls=[],
            correct_index=-1,
        )
        sess_log = log_root / f"session_{i:02d}_{ref.sub_task}_{inst.task_id}"
        r = run_agent_task(
            task=adapt, system=system, vlm=vlm, ceiling="full_pipeline",
            page=page, base_url=base_url, max_steps=max(20, len(inst.turns) + 5),
            log_dir=sess_log,
        )
        # Retention check at the START of this session.
        rmetrics = _retrieve_for_session(system, inst, k=5)
        session_results.append(SessionResult(
            task_id=inst.task_id,
            sub_task=inst.sub_task,
            order_index=i,
            correct=r.correct,
            final_url=r.final_url,
            final_wishlist=list(cumulative_wishlist),
            n_steps=r.n_steps,
            retrieval_top5=rmetrics["top_k"] or None,
            irr_at_5=rmetrics["irr_at_k"],
            mrr_at_5=rmetrics["mrr_at_k"],
            retrieval_set_hit=rmetrics["set_hit"],
            failure_mode=r.failure_mode,
        ))
        if r.correct and inst.ground_truth.target_variant_id:
            cumulative_wishlist.append(inst.ground_truth.target_variant_id)

    cumulative_ok = score_cumulative(task, cumulative_wishlist)
    all_ok = all(s.correct for s in session_results)
    return MultiSessionResult(
        task_id=task.task_id,
        variant=task.variant,
        sessions=session_results,
        correct=(all_ok and cumulative_ok),
        cumulative_correct=cumulative_ok,
        elapsed_ms=int((time.time() - t0) * 1000),
        notes=f"live, n_sessions={len(task.sessions)}",
    )


def _adapt_turns_to_phases(inst: TaskInstance) -> tuple[str, List[str], str]:
    """Best-effort adapter from a turn list to (anchor, fillers, recall_url).
    Used only by the live runner to reuse the 3-phase AgentTask loop.
    """
    agent_urls = [t.expected_url for t in inst.turns
                  if t.role == "agent" and t.expected_url]
    anchor_url = ""
    fillers: List[str] = []
    recall_url = ""
    if agent_urls:
        anchor_url = agent_urls[0]
        if len(agent_urls) > 2:
            fillers = agent_urls[1:-1]
            recall_url = agent_urls[-1]
        elif len(agent_urls) == 2:
            recall_url = agent_urls[-1]
        else:
            recall_url = agent_urls[0]
    return anchor_url, fillers, recall_url


def _recall_instruction(inst: TaskInstance) -> str:
    """Return the last non-empty user turn content; that's the recall cue."""
    for t in reversed(inst.turns):
        if t.role == "user" and t.content:
            return t.content
    return ""
