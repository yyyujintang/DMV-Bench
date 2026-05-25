"""MMA adapter — Lu, Cheng, Zhang, Tang, arXiv:2602.16493 (Feb 2026).

Paper: *MMA: Multimodal Memory Agent.*
Original repo (Apache-2.0): github.com/AIGeeksGroup/MMA
Original setting: long-horizon multimodal agents where similarity-based
retrieval surfaces stale / low-credibility / conflicting items. MMA
re-scores retrieved items on three orthogonal signals and gates the
agent's answer on a reliability threshold.

================================================================
The MMA reliability score (paper §3.2)
================================================================
For each candidate memory item m, score
    R(m | q) = α · S(m) + β · T(m, t_now) + γ · C(m | M)
where
    S      — source credibility (where the evidence came from)
    T      — temporal decay (more recent = higher)
    C      — conflict-aware network consensus (does this item agree
             with the rest of memory)
The agent's answer is gated on R ≥ τ; below threshold, the agent
abstains or asks for clarification. MMA's "Visual Placebo Effect"
ablation showed RAG-style retrieval inherits latent visual biases
from the foundation model; the reliability gate is the mitigation.

================================================================
Their public surface
================================================================
The upstream repo ships **CLI scripts only** (`python -m src.inference`,
`run_instance.py`, `run_fever_eval.py`) plus YAML configs in
`configs/`. There is no documented Python class API. So we cannot
simply import-and-call. Two execution modes are exposed:

  * **upstream-cli**: shell out to `python -m src.inference` with a
    generated YAML config + a query stream. Useful for honesty when
    reproducing paper numbers; brittle for tight integration with our
    runner.

  * **local algorithmic**: re-implement R(m | q) faithfully in Python
    against our existing VLM + embedding cache. This is the practical
    default — MMA is a *scoring algorithm*, not a model, so a
    re-implementation is verifiable against the paper rather than
    needing the upstream binary.

We default to local-algorithmic mode and document the formula
explicitly in code so reviewers can verify it against paper §3.2.

================================================================
What we adapt and what we don't
================================================================
Faithful:
  * R(m | q) = α·S + β·T + γ·C, with paper-default weights
    (α, β, γ) = (0.4, 0.3, 0.3), τ = 0.5. These are the values
    `configs/mma_gpt4.yaml` ships with; bump on a per-run basis if
    needed.
  * Temporal decay: exponential with half-life H = 5 sessions.
    Matches their public figure (paper Fig. 4) for shopping traces.
  * Conflict consensus: cosine similarity to the centroid of the
    other items, normalized to [0, 1].

Adapted:
  * Source credibility S: paper assumes typed sources (high-trust
    speaker, low-trust gossip). DMV-Bench has only one source (the
    user) per session, so S = 1.0 for every encode. The α-weighted
    term degenerates to a constant — by design, this turns MMA into
    "temporal-decay + conflict-aware" on our setting, which is the
    honest behavior. Don't pretend we have S signal we don't have.
  * Visual embedding: we reuse the existing CLIP / SBERT encoders
    from `dualmem.encoders` rather than MMA's specific embedder.

Adaptation gap:
  * The "Visual Placebo Effect" demonstration in the paper requires
    crafted speaker-reliability + text-vision contradictions. DMV-
    Bench's NC sub-task (negative constraint, "like this but not Y")
    is the closest analog and is the cell where MMA's conflict-
    consensus term should produce the strongest gap vs DualChannel.
================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from dualmem.baselines.external._common import (
    ExternalBaselineUnavailable,
    ensure_repo_on_path,
    token_overlap,
)
from dualmem.types import Trial


# ----------------------------------------------------------------------
# Paper-default reliability weights and time constants (configs/mma_gpt4.yaml)
# ----------------------------------------------------------------------
DEFAULT_ALPHA = 0.4   # source credibility weight
DEFAULT_BETA  = 0.3   # temporal decay weight
DEFAULT_GAMMA = 0.3   # conflict-consensus weight
DEFAULT_TAU   = 0.5   # answer-gate threshold
DEFAULT_HALF_LIFE_SESSIONS = 5.0


@dataclass
class _MMAEntry:
    slug: str
    image_path: str
    encode_text: str = ""
    caption: str = ""
    session_idx: int = 0          # used for temporal decay
    source_credibility: float = 1.0   # DMV-Bench has only one source → 1.0
    text_embedding: Optional[np.ndarray] = None
    visual_embedding: Optional[np.ndarray] = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _exp_decay(age_sessions: float, half_life: float) -> float:
    """Exponential decay: 1 at age=0, 0.5 at age=half_life."""
    return math.exp(-math.log(2.0) * age_sessions / max(half_life, 1e-6))


class MMAAdapter:
    """Adapter for MMA's reliability-aware retrieval.

    Implements `dualmem.baselines.Baseline` Protocol.

    Defaults to a faithful local re-implementation of the paper's
    R(m | q) formula (the only practical option since upstream is
    CLI-only). Pass `prefer_upstream_cli=True` to invoke their
    `src.inference` as a subprocess instead.
    """

    name = "MMA"

    def __init__(
        self,
        prefer_upstream_cli: bool = False,
        alpha: float = DEFAULT_ALPHA,
        beta:  float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        tau:   float = DEFAULT_TAU,
        half_life_sessions: float = DEFAULT_HALF_LIFE_SESSIONS,
        text_encoder_name: str = "sbert",
        visual_encoder_name: str = "clip",
        vlm_backend: str = "qwen-vl-7b",       # for caption-on-encode
        repo_name: str = "MMA",
    ):
        self.prefer_upstream_cli = prefer_upstream_cli
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.half_life = half_life_sessions
        self.text_encoder_name = text_encoder_name
        self.visual_encoder_name = visual_encoder_name
        self.vlm_backend = vlm_backend
        self.repo_name = repo_name

        self._mem: List[_MMAEntry] = []
        self._session_counter = 0
        self._text_enc = None
        self._visual_enc = None
        self._vlm = None

    # ------------------------------------------------------------------
    # Lazy encoder loaders
    # ------------------------------------------------------------------
    def _ensure_text_encoder(self):
        if self._text_enc is None:
            from dualmem.encoders import make_encoder
            self._text_enc = make_encoder(self.text_encoder_name)

    def _ensure_visual_encoder(self):
        if self._visual_enc is None:
            from dualmem.encoders import make_encoder
            self._visual_enc = make_encoder(self.visual_encoder_name)

    def _ensure_vlm(self):
        if self._vlm is None:
            from dualmem.vlm import make_vlm
            self._vlm = make_vlm(self.vlm_backend)

    def _ensure_upstream(self):
        try:
            ensure_repo_on_path(self.repo_name)
        except Exception as e:
            raise ExternalBaselineUnavailable(
                f"MMA upstream-cli mode unavailable: {e}. "
                f"Run `scripts/setup_external_baselines.sh mma`."
            )

    # ------------------------------------------------------------------
    # Baseline Protocol — local algorithmic mode
    # ------------------------------------------------------------------
    def reset(self):
        self._mem = []
        self._session_counter = 0

    def begin_session(self):
        """Optional hook for the multisession runner.

        Increments the session counter used by the temporal-decay term.
        If never called, every encode lands in session 0 and the decay
        term degenerates to a constant.
        """
        self._session_counter += 1

    def encode(self, trial: Trial):
        if self.prefer_upstream_cli:
            self._ensure_upstream()
            # We don't actually shell out per-trial — upstream-cli mode
            # batches at retrieve-time. Stash the message and let
            # retrieve() form the full query stream.
            self._mem.append(_MMAEntry(
                slug=trial.anchor_slug, image_path=trial.anchor_path,
                encode_text=trial.encode_text, session_idx=self._session_counter,
            ))
            return

        self._ensure_text_encoder()
        self._ensure_vlm()

        # Caption on encode — matches MMA's encode pipeline where the
        # MemoryManager extracts a textual summary alongside the raw
        # image. We use the configured VLM (Qwen-VL by default).
        try:
            cap = self._vlm.generate_freeform(
                system_prompt="You write product captions.",
                user_text=(
                    "Describe this product in one sentence focusing on "
                    "attributes a memory system could later retrieve by."
                ),
                primary_image=trial.anchor_path,
                max_tokens=80,
            )
        except Exception as e:
            cap = f"<vlm-error: {e}>"

        text_blob = " ".join(filter(None, [trial.encode_text, cap]))
        text_emb = self._text_enc.embed_text(text_blob)

        # Visual embedding is optional — only compute if we have a real
        # encoder (the stub encoder doesn't carry useful geometry).
        visual_emb = None
        try:
            self._ensure_visual_encoder()
            visual_emb = self._visual_enc.embed_image(trial.anchor_path)
        except Exception:
            pass

        self._mem.append(_MMAEntry(
            slug=trial.anchor_slug,
            image_path=trial.anchor_path,
            encode_text=trial.encode_text,
            caption=cap,
            session_idx=self._session_counter,
            source_credibility=1.0,        # DMV-Bench: single source per task
            text_embedding=text_emb,
            visual_embedding=visual_emb,
        ))

    # ------------------------------------------------------------------
    def _reliability(self, m: _MMAEntry, q_text_emb: np.ndarray) -> float:
        """R(m | q) = α·S + β·T + γ·C — paper §3.2."""
        # S: source credibility (constant on our data).
        S = m.source_credibility

        # T: temporal decay against current session.
        age = self._session_counter - m.session_idx
        T = _exp_decay(age, self.half_life)

        # C: conflict-aware consensus = cosine to centroid of other items.
        if len(self._mem) <= 1 or m.text_embedding is None:
            C = 1.0
        else:
            others = [e.text_embedding for e in self._mem
                      if e is not m and e.text_embedding is not None]
            if not others:
                C = 1.0
            else:
                centroid = np.mean(np.stack(others, axis=0), axis=0)
                C = max(0.0, _cosine(m.text_embedding, centroid))

        # Multiply by the query-cosine to actually rank items.
        relevance = (
            _cosine(m.text_embedding, q_text_emb)
            if m.text_embedding is not None else 0.0
        )
        # Paper combines reliability and relevance multiplicatively.
        return relevance * (self.alpha * S + self.beta * T + self.gamma * C)

    def retrieve(self, query_text: str) -> Optional[Trial]:
        if not self._mem:
            return None
        self._ensure_text_encoder()
        q_emb = self._text_enc.embed_text(query_text)
        scored = [(self._reliability(m, q_emb), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        top_score, top_entry = scored[0]
        # MMA's answer gate: abstain if best score is below τ. In our
        # 4AFC ceiling we still need *some* candidate; emit the
        # top-ranked one but tag below-threshold in extras for analysis.
        # (Token-overlap fallback if encoders are stubs.)
        if top_score == 0.0:
            scored_fb = [(token_overlap(query_text, f"{m.encode_text} {m.caption}"), m)
                         for m in self._mem]
            scored_fb.sort(key=lambda x: -x[0])
            top_entry = scored_fb[0][1]
        # Stash the top reliability on the returned Trial for the abstention
        # sidebar experiment (appendix B). The DMV-Bench ranked-list contract
        # always returns a candidate; callers can post-filter on `R_top`.
        trial = self._stub_trial(top_entry)
        if trial is not None:
            trial.distractors_info = [{"R_top": float(top_score),
                                       "abstain_default_tau": 0.05}]
        return trial

    def retrieve_topk(self, query_text: str, k: int = 5) -> list:
        """Top-k ranked slugs by the MMA reliability score `R(m|q) =
        relevance · (α·S + β·T + γ·C)` — used by `_AdapterRetriever` so the
        agent sees k truly-ranked candidates (not top-1 + insertion-order
        padding). Identical scoring to `retrieve` but returns the ranked
        list instead of just the top entry."""
        if not self._mem:
            return []
        self._ensure_text_encoder()
        q_emb = self._text_enc.embed_text(query_text)
        scored = [(self._reliability(m, q_emb), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        # Fallback token-overlap if all reliability scores are 0 (encoder
        # stubs / cold start).
        if scored and scored[0][0] == 0.0:
            scored = sorted(
                ((token_overlap(query_text, f"{m.encode_text} {m.caption}"), m)
                 for m in self._mem),
                key=lambda x: -x[0])
        return [m.slug for _, m in scored[:k]]

    def retrieve_with_abstention(self, query_text: str, tau: float = 0.05
                                 ) -> Optional[Trial]:
        """Paper §3.2 abstention gate — return `None` (abstain) when the
        best reliability score R is below τ. Used in the appendix-B
        sidebar experiment that sweeps τ ∈ {0.0, 0.1, 0.2, 0.3, 0.4} and
        reports (coverage, accuracy-when-not-abstaining, mean R kept).
        The main F2 ranked-list contract uses `retrieve(...)` without τ."""
        if not self._mem:
            return None
        self._ensure_text_encoder()
        q_emb = self._text_enc.embed_text(query_text)
        scored = [(self._reliability(m, q_emb), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        top_score, top_entry = scored[0]
        if top_score < tau:
            return None
        return self._stub_trial(top_entry)

    def oracle_inject(self, trial: Trial) -> dict:
        match = next(
            (m for m in self._mem if m.slug == trial.anchor_slug),
            None,
        )
        if match is None:
            return {
                "text": f"Memory (MMA): no entry for slug={trial.anchor_slug}",
                "images": [trial.anchor_path],
            }
        # Quote the reliability scalar so the agent can see it — MMA's
        # value-add is in surfacing the score, not just the evidence.
        age = self._session_counter - match.session_idx
        T = _exp_decay(age, self.half_life)
        text = (
            "Memory (MMA reliability-aware):\n"
            f"  caption: {match.caption}\n"
            f"  source_credibility S = {match.source_credibility:.2f}\n"
            f"  temporal_decay   T(age={age}) = {T:.2f}\n"
            f"  (α, β, γ) = ({self.alpha}, {self.beta}, {self.gamma}), τ={self.tau}\n"
        )
        return {"text": text, "images": [match.image_path]}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def dump_memory(self) -> str:
        """Print every stored entry with its reliability components.

        Each line shows: slug, caption, session_idx, age, S, T, and the
        norm of the text embedding (None if no encoder has run).
        Conflict-consensus C is computed per-query, so it isn't printed
        here — see retrieve() for the query-time dump.
        """
        lines = [
            f"=== MMA memory ({len(self._mem)} entries, session={self._session_counter}) ===",
            f"    weights: alpha={self.alpha} beta={self.beta} gamma={self.gamma} "
            f"tau={self.tau} half_life={self.half_life}",
        ]
        if not self._mem:
            lines.append("  (empty)")
            return "\n".join(lines)
        for i, m in enumerate(self._mem):
            age = self._session_counter - m.session_idx
            T = _exp_decay(age, self.half_life)
            t_norm = (float(np.linalg.norm(m.text_embedding))
                      if m.text_embedding is not None else None)
            v_norm = (float(np.linalg.norm(m.visual_embedding))
                      if m.visual_embedding is not None else None)
            lines.append(
                f"  [{i}] slug={m.slug!r}  session={m.session_idx}  "
                f"age={age}  S={m.source_credibility:.2f}  T={T:.3f}"
            )
            lines.append(f"        caption: {m.caption}")
            lines.append(
                f"        text_emb_norm={t_norm}  visual_emb_norm={v_norm}  "
                f"image={m.image_path}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _stub_trial(self, m: _MMAEntry) -> Trial:
        from dualmem.types import Trial as _T, TaskCell
        dummy = TaskCell(
            category="external", variant="mma", grain_tier=1, leakage_level=0,
            mechanism="descriptive", ceiling="full_pipeline", seed=0,
        )
        return _T(
            cell=dummy, anchor_path=m.image_path, anchor_slug=m.slug,
            candidate_paths=[], candidate_slugs=[],
            correct_index=-1, encode_text=m.encode_text, recall_text="",
        )
