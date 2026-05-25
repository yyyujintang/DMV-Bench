"""WorldMM adapter — Yeo et al., CVPR 2026 Highlight, arXiv:2512.02425.

Paper: *WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning.*

Original repo (Apache-2.0): github.com/wgcyeo/WorldMM
Original setting: long-form video reasoning over EgoLifeQA + Video-MME,
with a 3-module memory split (episodic / semantic / visual) and adaptive
retrieval across modules.

================================================================
What we adapt and what we don't
================================================================

Faithful:
  * Three parallel memory modules: episodic / semantic / visual.
  * Adaptive retrieval: query is routed to the module that best matches
    the query's modality cue (date/time-like → episodic, abstract
    concept → semantic, visual descriptor → visual).
  * Their evaluation prompt templates from `script/4_eval.sh` are reused
    verbatim where applicable.

Adapted:
  * No video. Each DMV-Bench *session* is treated as one "video segment"
    — the segment's frames are the per-turn screenshots, the segment's
    timestamp is the session index.
  * Backbone swap: their `script/4_eval.sh` defaults are GPT-5-mini
    (retriever) + GPT-5 (responder). We allow `--retriever-vlm` and
    `--responder-vlm` to be set to any backbone in `dualmem.vlm`. Default
    here is Qwen2.5-VL-7B for both, matching our open-source slot.
  * Their data ingestion expects EgoLifeQA-style JSONL (timestamps,
    speaker IDs, etc.). We bypass that and feed Trials directly into the
    three modules via `encode()`.

Adaptation gap: their adaptive-retrieval router is a learned policy
fine-tuned on EgoLifeQA. We use a zero-shot heuristic router (keyword
match against module-specific cue lists). On the EgoLifeQA-vs-shopping
shift this is the most uncertain part of the adaptation.

WorldMM's 3 modules map cleanly onto DMV-Bench's 5 task types:
  episodic ↔ PD (preference drift, time-ordered)
  semantic ↔ SA, NC (style / negation constraints — abstract)
  visual   ↔ VL, IC (landmark / incidental cue — pictorial)
This gives us a clean ablation hook: knock out one module, expect the
corresponding task type to drop.
================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dualmem.baselines.external._common import (
    ExternalBaselineUnavailable,
    ensure_repo_on_path,
    token_overlap,
)
from dualmem.types import Trial


# Heuristic cue lists for the router. Drawn from WorldMM's appendix
# § A.3 ("module-prompting keywords"), trimmed to e-commerce vocab.
ROUTER_CUES = {
    "episodic": {
        "earlier", "before", "first", "last", "previous", "session",
        "after", "now", "still", "any more", "remember when",
    },
    "semantic": {
        "like", "similar", "style", "kind", "category", "type",
        "not", "no", "without", "instead", "rather", "prefer",
    },
    "visual": {
        "color", "red", "blue", "green", "shape", "pattern",
        "wood", "metal", "fabric", "leather", "image", "looks", "looking",
    },
}


@dataclass
class _Segment:
    """One ingested 'video segment' = one DMV-Bench session's anchor."""
    slug: str
    image_path: str
    episodic_text: str = ""     # timestamps + ordering ("session 2, turn 5")
    semantic_text: str = ""     # abstract attributes
    visual_text: str = ""       # caption focused on pictorial features


class WorldMMAdapter:
    """Adapter for WorldMM's tri-modular memory + adaptive retrieval."""

    name = "WorldMM"

    def __init__(
        self,
        retriever_vlm: str = "qwen-vl-7b",
        responder_vlm: str = "qwen-vl-7b",
        repo_name: str = "WorldMM",
    ):
        self.retriever_vlm_backend = retriever_vlm
        self.responder_vlm_backend = responder_vlm
        self.repo_name = repo_name
        self._segments: Dict[str, _Segment] = {}
        self._order: List[str] = []         # insertion order = "session index"
        self._vlm = None

    # ------------------------------------------------------------------
    def _ensure_vlm(self):
        if self._vlm is not None:
            return
        from dualmem.vlm import make_vlm
        # Use retriever backbone for memory-construction prompts (matches
        # WorldMM's `script/3_build_memory.sh --retriever-model`).
        self._vlm = make_vlm(self.retriever_vlm_backend)

    def _ensure_repo(self):
        try:
            ensure_repo_on_path(self.repo_name)
        except Exception as e:
            raise ExternalBaselineUnavailable(
                f"WorldMM repo unavailable: {e}. "
                f"Run `scripts/setup_external_baselines.sh worldmm`."
            )

    # ------------------------------------------------------------------
    # Module-specific extractors. Prompts are derived from WorldMM's
    # `script/3_build_memory.sh` step prompts (episodic / semantic /
    # visual). Each pass returns a short text block.
    # ------------------------------------------------------------------
    _EPISODIC_PROMPT = (
        "Summarize when and how this product appeared in the user's browsing "
        "history. Use the form 'Session N, turn M: <one-sentence event>'."
    )
    _SEMANTIC_PROMPT = (
        "List the abstract attributes that define this product's category and "
        "style: 1-2 short clauses, no visual details, no numbers."
    )
    _VISUAL_PROMPT = (
        "List only the visual attributes of this product: color, material, "
        "shape, distinctive marks. One sentence, comma-separated."
    )

    def _build_module_text(self, prompt: str, trial: Trial) -> str:
        self._ensure_vlm()
        try:
            return self._vlm.generate_freeform(
                system_prompt="You write concise memory entries.",
                user_text=prompt,
                primary_image=trial.anchor_path,
                max_tokens=120,
            )
        except Exception as e:
            return f"<vlm-error: {e}>"

    # ------------------------------------------------------------------
    # Baseline Protocol
    # ------------------------------------------------------------------
    def reset(self):
        self._segments = {}
        self._order = []
        self._slug_to_session: Dict[str, int] = {}
        self._cur_session: int = 0

    def set_session(self, s: int) -> None:
        """Hook called by the F2 runner BEFORE replaying a session's trajectory;
        lets the multi-grain temporal index attribute encodes to a session_idx
        (the paper's 'multi-granularity time axis')."""
        self._cur_session = int(s)

    def encode(self, trial: Trial):
        slug = trial.anchor_slug
        if slug in self._segments:
            return
        seg = _Segment(slug=slug, image_path=trial.anchor_path)
        # Episodic: light synth (no VLM call) — just timestamped placement.
        seg.episodic_text = (
            f"Session {self._cur_session}, turn {trial.filler_steps}: "
            f"user viewed {slug}."
        )
        # Semantic + visual: one VLM call each.
        seg.semantic_text = self._build_module_text(self._SEMANTIC_PROMPT, trial)
        seg.visual_text = self._build_module_text(self._VISUAL_PROMPT, trial)
        self._segments[slug] = seg
        self._order.append(slug)
        # also remember which session this product was encoded in — drives
        # the per-session/multi-session grains in retrieval.
        if not hasattr(self, "_slug_to_session"):
            self._slug_to_session = {}
        self._slug_to_session[slug] = self._cur_session

    # ------------------------------------------------------------------
    # Adaptive router
    # ------------------------------------------------------------------
    @staticmethod
    def route(query_text: str) -> str:
        """Return 'episodic' | 'semantic' | 'visual' based on cue overlap."""
        ql = query_text.lower()
        scores = {
            module: sum(1 for c in cues if re.search(rf"\b{re.escape(c)}\b", ql))
            for module, cues in ROUTER_CUES.items()
        }
        # Ties broken by precedence visual > semantic > episodic (matches
        # WorldMM's tie-break order for grounding-heavy queries).
        for m in ("visual", "semantic", "episodic"):
            if scores[m] == max(scores.values()) and scores[m] > 0:
                return m
        return "semantic"  # default if nothing matches

    # ------------------------------------------------------------------
    # Multi-granularity temporal index (paper §3.2 — adapted to DMV-Bench).
    # ------------------------------------------------------------------
    # The paper indexes captions at 5 time-scales (10s/30s/3min/10min/1h);
    # for DMV-Bench's session-structured browse we adapt to 3 grains —
    # per-product (finest, gives the slug), per-session (mid), per-chain
    # (coarsest). At retrieve time each grain contributes a per-slug score;
    # the final rank is a weighted sum (paper-style multi-scale fusion).
    _GRAIN_WEIGHTS = {"product": 0.55, "session": 0.30, "chain": 0.15}

    def _grain_texts(self, module: str) -> tuple[dict, dict, str]:
        """Return (per-product text, per-session aggregated text, chain
        aggregated text) for the chosen module."""
        attr = f"{module}_text"
        per_product = {slug: getattr(seg, attr) for slug, seg in self._segments.items()}
        per_session: dict = {}
        for slug, seg in self._segments.items():
            sess = getattr(self, "_slug_to_session", {}).get(slug, 0)
            per_session.setdefault(sess, []).append(getattr(seg, attr))
        per_session_agg = {s: " ".join(xs) for s, xs in per_session.items()}
        chain_agg = " ".join(per_product.values())
        return per_product, per_session_agg, chain_agg

    # Multi-round adaptive routing budget (paper's MAX_ROUNDS=5; we use 3
    # to balance fidelity against API cost for DMV-Bench scale).
    _MAX_ROUNDS = 3

    def _module_scores(self, query_text: str, module: str) -> dict:
        """Multi-grain weighted score per slug, for ONE module."""
        per_product, per_session_agg, chain_agg = self._grain_texts(module)
        s1 = {s: token_overlap(query_text, t) for s, t in per_product.items()}
        s2_session = {sess: token_overlap(query_text, agg)
                      for sess, agg in per_session_agg.items()}
        s2 = {s: s2_session.get(
                  getattr(self, "_slug_to_session", {}).get(s, 0), 0)
              for s in per_product}
        s3_score = token_overlap(query_text, chain_agg)
        w = self._GRAIN_WEIGHTS
        return {s: w["product"] * s1[s] + w["session"] * s2[s]
                   + w["chain"] * s3_score
                for s in per_product}

    def _adaptive_scores(self, query_text: str) -> dict:
        """Run the adaptive multi-round routing and return the accumulated
        per-slug score dict (used by both `retrieve` for top-1 and
        `retrieve_topk` for ranked top-k)."""
        if not self._segments:
            return {}

        queried: list = []                              # modules already queried
        combined: dict = {}                             # slug -> accumulated score
        ctx_summary: list = []                          # short history for the VLM

        for r in range(self._MAX_ROUNDS):
            if r == 0:
                module = self.route(query_text)         # keyword router, no VLM
            else:
                self._ensure_vlm()
                prompt = (
                    f"Retrieval task. Query: {query_text!r}\n"
                    f"Modules already queried: {queried}\n"
                    f"Top candidates so far: " + " | ".join(ctx_summary[-3:]) + "\n"
                    f"Reply with the NEXT memory module to query from "
                    f"[episodic, semantic, visual], or 'done' if the evidence "
                    f"so far is sufficient.\nFormat: 'NEXT: <module-or-done>'"
                )
                try:
                    resp = self._vlm.generate_freeform(
                        system_prompt="You are a memory-routing controller.",
                        user_text=prompt, max_tokens=20)
                    m = re.search(r"NEXT:\s*(\w+)", resp.lower())
                    module = m.group(1) if m else "done"
                except Exception:
                    module = "done"

            if module == "done" or module in queried \
                    or module not in ("episodic", "semantic", "visual"):
                break
            queried.append(module)

            ms = self._module_scores(query_text, module)
            for s, sc in ms.items():
                combined[s] = combined.get(s, 0.0) + sc
            top3 = sorted(ms.items(), key=lambda x: -x[1])[:3]
            ctx_summary.append(f"[{module}] "
                               + ", ".join(f"{s}={sc:.1f}" for s, sc in top3))

        if not combined:
            # All modules said 'done' before retrieving anything — fall back
            # to a single-shot keyword-routed query so we still return something.
            ms = self._module_scores(query_text, self.route(query_text))
            combined = ms
        return combined

    def retrieve(self, query_text: str) -> Optional[Trial]:
        """Paper §3.5 adaptive multi-round routing — top-1 result."""
        combined = self._adaptive_scores(query_text)
        if not combined:
            return None
        best = max(combined, key=combined.get)
        return self._stub_trial(best)

    def retrieve_topk(self, query_text: str, k: int = 5) -> list:
        """Top-k ranked slugs by the adaptive-router combined score —
        used by `_AdapterRetriever` so the agent sees k truly-ranked
        candidates (not top-1 + insertion-order junk)."""
        combined = self._adaptive_scores(query_text)
        if not combined:
            return []
        ranked = sorted(combined.items(), key=lambda x: -x[1])
        return [slug for slug, _ in ranked[:k]]

    def oracle_inject(self, trial: Trial) -> dict:
        seg = self._segments.get(trial.anchor_slug)
        if seg is None:
            seg = _Segment(slug=trial.anchor_slug, image_path=trial.anchor_path)
        text = (
            "Memory (WorldMM tri-modular):\n"
            f"  episodic: {seg.episodic_text}\n"
            f"  semantic: {seg.semantic_text}\n"
            f"  visual:   {seg.visual_text}\n"
        )
        return {"text": text, "images": [seg.image_path]}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def dump_memory(self) -> str:
        """Human-readable snapshot of every stored segment, by module.

        Use this in smoke tests / debug runs to verify what each of the
        three WorldMM modules has captured for each anchor.
        """
        lines = [f"=== WorldMM memory ({len(self._segments)} segment(s)) ==="]
        if not self._segments:
            lines.append("  (empty)")
            return "\n".join(lines)
        for i, slug in enumerate(self._order):
            seg = self._segments[slug]
            lines.append(f"[{i}] slug={seg.slug!r}  image={seg.image_path}")
            lines.append(f"     episodic: {seg.episodic_text}")
            lines.append(f"     semantic: {seg.semantic_text}")
            lines.append(f"     visual:   {seg.visual_text}")
        return "\n".join(lines)

    def _stub_trial(self, slug: str) -> Optional[Trial]:
        seg = self._segments.get(slug)
        if seg is None:
            return None
        from dualmem.types import Trial as _T, TaskCell
        dummy_cell = TaskCell(
            category="external", variant="worldmm", grain_tier=1, leakage_level=0,
            mechanism="descriptive", ceiling="full_pipeline", seed=0,
        )
        return _T(
            cell=dummy_cell,
            anchor_path=seg.image_path,
            anchor_slug=seg.slug,
            candidate_paths=[],
            candidate_slugs=[],
            correct_index=-1,
            encode_text="",
            recall_text="",
        )
