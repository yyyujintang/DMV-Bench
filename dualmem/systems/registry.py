"""System factory: composes (encoder, bank, retriever, injector) into the
named memory architectures.

A memory architecture is exactly three choices -- what the bank stores
(ENCODE), how a recall query is matched against it (RETRIEVE), and what is
re-presented to the VLM (INJECT). Every system below is a point in that space,
so a gap between two of them is attributable to the axis they differ on.

The seven architectures reported in the paper:

| System        | Encode              | Retrieve                 | Inject        |
|---------------|---------------------|--------------------------|---------------|
| NoMemory      | (none)              | (none)                   | (none)        |
| TextOnly      | product-class text  | SBERT cosine             | text          |
| Caption       | VLM caption         | SBERT cosine             | caption       |
| WorldMM-lite  | episodic+sem+visual | adaptive iterative       | retrieved ctx |
| MMA-lite      | items + reliability | reliability-weighted     | text          |
| M2A-lite      | raw + semantic      | dense + BM25 + visual    | text          |
| DualMem       | image + caption     | hybrid (SigLIP-2, SBERT) | image+caption |

`DualMem-a75` (alpha=0.75, visual-dominant fusion) is the headline
configuration; the other `DualMem-*` names vary one axis each for the ablation
table. `DualChannel*` are the intermediate variants used to decompose the
DualMem-vs-M2A gap into fusion-rule and encoder effects.

`make_system(name, vlm_caption_fn=...)` returns a fully-constructed system.
Pass `text_encoder_name="stub"` / `visual_encoder_name="stub"` in tests to
avoid model downloads.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from dualmem.encoders import make_encoder
from dualmem.injection import (
    TextOnlyInject, CaptionOnlyInject, ImageOnlyInject, ImageTextInject,
)
from dualmem.memory import VisualBank, VerbalBank, DualBank
from dualmem.retrieval import (
    TextRetriever, VisualRetriever, HybridRetriever, MostRecentRetriever,
)
from dualmem.systems.base import MemorySystem


def _build(name: str, *,
           text_encoder_name: str = "sbert",
           visual_encoder_name: Optional[str] = "clip",
           vlm_caption_fn: Optional[Callable] = None) -> MemorySystem:

    if name == "TextOnly":
        sbert = make_encoder(text_encoder_name)
        bank = VerbalBank(sbert, mode="encode_text")
        return MemorySystem(
            name="TextOnly",
            bank=bank,
            retriever=TextRetriever(sbert),
            injector=TextOnlyInject(),
            caption_fn=None,
        )

    if name == "Caption":
        sbert = make_encoder(text_encoder_name)
        bank = VerbalBank(sbert, mode="vlm_caption")
        return MemorySystem(
            name="Caption",
            bank=bank,
            retriever=TextRetriever(sbert),
            injector=CaptionOnlyInject(),
            caption_fn=vlm_caption_fn,
        )

    if name in ("DualChannel", "DualChannel-norm", "DualChannel-siglip"):
        # The DualChannel family — three single-knob variants used to
        # decompose the DualChannel vs M2A-lite gap into three orthogonal
        # axes (see doc/m2a_vs_dualchannel.md):
        #   - DualChannel        — RRF fusion, CLIP visual encoder.
        #   - DualChannel-norm   — min-max sim fusion, CLIP visual encoder.
        #   - DualChannel-siglip — min-max sim fusion, siglip2 visual encoder.
        # Δ(DualChannel-norm − DualChannel) isolates the fusion-rule effect.
        # Δ(DualChannel-siglip − DualChannel-norm) isolates the encoder effect.
        # Δ(M2A-lite − DualChannel-siglip) isolates the BM25 / 3-way effect.
        from dualmem.retrieval.hybrid_retriever import HybridNormRetriever
        sbert = make_encoder(text_encoder_name)
        vis_name = "siglip2" if name == "DualChannel-siglip" else (visual_encoder_name or "clip")
        clip_v = make_encoder(vis_name) if vis_name else None
        bank = DualBank(visual_encoder=clip_v, text_encoder=sbert, verbal_mode="vlm_caption")
        text_r = TextRetriever(sbert)
        if clip_v is not None:
            visual_r = VisualRetriever(clip_v, query_mode="text")
            if name == "DualChannel":
                retriever = HybridRetriever(visual=visual_r, textual=text_r, alpha=0.5)
            else:   # DualChannel-norm / DualChannel-siglip
                retriever = HybridNormRetriever(visual=visual_r, textual=text_r, alpha=0.5)
        else:
            retriever = text_r
        return MemorySystem(
            name=name,
            bank=bank,
            retriever=retriever,
            injector=ImageTextInject(),
            caption_fn=vlm_caption_fn,
        )

    if name in ("DualMem", "DualMem-vis", "DualMem-verb",
                "DualMem-img", "DualMem-cap",
                "DualMem-a25", "DualMem-a75",
                "DualMem-vis-img", "DualMem-vis-cap"):
        # DualMem ablation variants — table 'DualMem ablations' in the paper.
        # All hold the base configuration fixed (= DualChannel-siglip: SigLIP-2
        # visual encoder + SBERT text encoder + DualBank + HybridNormRetriever
        # (α=0.5, min-max sim fusion) + ImageTextInject) and vary ONE axis:
        #   DualMem       — alias of the base (full system, all knobs on).
        #   DualMem-vis   — visual-only retrieval (drop the verbal channel
        #                   at retrieval; verbal still encoded into bank so
        #                   captions remain available for injection).
        #   DualMem-verb  — verbal-only retrieval (drop the visual channel
        #                   at retrieval).
        #   DualMem-img   — image-only injection (drop the caption text from
        #                   what gets re-shown to the VLM).
        #   DualMem-cap   — caption-only injection (drop the memory image).
        from dualmem.retrieval.hybrid_retriever import HybridNormRetriever
        sbert = make_encoder(text_encoder_name)
        clip_v = make_encoder("siglip2")
        bank = DualBank(visual_encoder=clip_v, text_encoder=sbert,
                        verbal_mode="vlm_caption")
        text_r = TextRetriever(sbert)
        visual_r = VisualRetriever(clip_v, query_mode="text")

        # retrieval axis
        if name in ("DualMem-vis", "DualMem-vis-img", "DualMem-vis-cap"):
            # visual-only retrieval (alpha=1.0 effectively)
            retriever = visual_r
        elif name == "DualMem-verb":
            # verbal-only retrieval (alpha=0.0)
            retriever = text_r
        elif name == "DualMem-a25":
            retriever = HybridNormRetriever(visual=visual_r, textual=text_r,
                                            alpha=0.25)
        elif name == "DualMem-a75":
            retriever = HybridNormRetriever(visual=visual_r, textual=text_r,
                                            alpha=0.75)
        else:
            # DualMem (full), DualMem-img, DualMem-cap → α=0.5 default hybrid
            retriever = HybridNormRetriever(visual=visual_r, textual=text_r,
                                            alpha=0.5)

        # injection axis
        if name in ("DualMem-img", "DualMem-vis-img"):
            injector = ImageOnlyInject()
        elif name in ("DualMem-cap", "DualMem-vis-cap"):
            injector = CaptionOnlyInject()
        else:
            injector = ImageTextInject()

        return MemorySystem(
            name=name, bank=bank, retriever=retriever, injector=injector,
            caption_fn=vlm_caption_fn,
        )

    if name == "CLIPVision":
        # NOTE: this used to be called "CoMEM" but the name collided with the
        # CoMEM-Agent paper (Wu et al. 2025, arXiv:2510.09038) which we now
        # ship as an external baseline (see dualmem/baselines/external/).
        # Internally this is just CLIP-image-only retrieval + image injection.
        clip = make_encoder(visual_encoder_name)
        bank = VisualBank(encoder=clip)
        return MemorySystem(
            name="CLIPVision",
            bank=bank,
            retriever=VisualRetriever(clip, query_mode="text"),
            injector=ImageOnlyInject(),
            caption_fn=None,
        )

    if name == "HYMEM":
        # HSE-Mem-style hybrid: caption-residual + visual residual.
        # Differs from DualChannel: text-DOMINANT hybrid retrieval (alpha=0.3,
        # caption drives retrieval, visual residual confirms) AND image-only
        # injection (the discrete caption is for retrieval, the continuous
        # visual residual is what gets re-shown to the VLM). This matches the
        # HSE-Mem motivation: hippocampal/discrete index + neocortical/continuous
        # representation. In our black-box setting we approximate "continuous
        # tokens" by the raw image.
        sbert = make_encoder(text_encoder_name)
        clip = make_encoder(visual_encoder_name)
        bank = DualBank(visual_encoder=clip, text_encoder=sbert, verbal_mode="vlm_caption")
        visual_r = VisualRetriever(clip, query_mode="text")
        text_r = TextRetriever(sbert)
        return MemorySystem(
            name="HYMEM",
            bank=bank,
            retriever=HybridRetriever(visual=visual_r, textual=text_r, alpha=0.3),
            injector=ImageOnlyInject(),
            caption_fn=vlm_caption_fn,
        )

    if name == "NoMemory":
        # Absolute floor — bank resets at every encode call, so the bank is
        # always empty when retrieval is queried. Equivalent to "agent has
        # no external memory whatsoever". Used as paper §4 lower bound.
        sbert = make_encoder(text_encoder_name)
        bank = _EphemeralVerbalBank(sbert)
        return MemorySystem(
            name="NoMemory",
            bank=bank,
            retriever=TextRetriever(sbert),
            injector=TextOnlyInject(),
            caption_fn=None,
        )

    if name == "LongContext":
        # Stores raw text of every encoded page (no embedding, no scoring).
        # At retrieval, returns ALL stored entries — the runner concatenates
        # them into the VLM prompt verbatim. Matches the "long-context only"
        # baseline of MemoryArena: no external memory, just cram prior
        # session traces into the VLM context window.
        sbert = make_encoder(text_encoder_name)
        bank = VerbalBank(sbert, mode="encode_text")
        return MemorySystem(
            name="LongContext",
            bank=bank,
            retriever=_FullDumpRetriever(),
            injector=TextOnlyInject(),
            caption_fn=None,
        )

    raise KeyError(f"Unknown system {name!r}")


class _EphemeralVerbalBank:
    """Bank that drops every entry as soon as it's written. For NoMemory baseline."""
    name = "ephemeral"
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder
        self._entries: list = []
    def reset(self): self._entries = []
    def encode(self, image_path, slug, *, encode_text="", caption=None):
        from dualmem.memory.entry import MemoryEntry
        return MemoryEntry(slug=slug, image_path=image_path, encode_text=encode_text, caption=caption)
    def entries(self): return []
    def find_by_slug(self, slug): return None


class _FullDumpRetriever:
    """Returns ALL bank entries unscored — caller injects them all into VLM context."""
    name = "full_dump"
    def retrieve(self, query, entries, k=1):
        return list(entries)   # ignore k; caller wants everything


def _ext(name: str, **kw):
    """Defer the import — external_wrap pulls in the external adapters,
    which can be heavy.
    Forwards `vlm_caption_fn` so `-text` variants can carry the F2 caption."""
    from dualmem.systems.external_wrap import make_external_system
    return make_external_system(name, vlm_caption_fn=kw.get("vlm_caption_fn"))


SYSTEM_REGISTRY: Dict[str, Callable[..., MemorySystem]] = {
    "NoMemory":     lambda **kw: _build("NoMemory", **kw),
    "LongContext":  lambda **kw: _build("LongContext", **kw),
    "TextOnly":     lambda **kw: _build("TextOnly", **kw),
    "Caption":      lambda **kw: _build("Caption", **kw),
    "DualChannel":        lambda **kw: _build("DualChannel", **kw),
    "DualChannel-norm":   lambda **kw: _build("DualChannel-norm", **kw),
    "DualChannel-siglip": lambda **kw: _build("DualChannel-siglip", **kw),
    # DualMem ablation table (paper §4): one knob per row.
    "DualMem":            lambda **kw: _build("DualMem", **kw),
    "DualMem-vis":        lambda **kw: _build("DualMem-vis", **kw),
    "DualMem-verb":       lambda **kw: _build("DualMem-verb", **kw),
    "DualMem-img":        lambda **kw: _build("DualMem-img", **kw),
    "DualMem-cap":        lambda **kw: _build("DualMem-cap", **kw),
    # Extra α-sweep points (DualMem is α=0.5, DualMem-vis = 1.0, DualMem-verb = 0.0).
    "DualMem-a25":        lambda **kw: _build("DualMem-a25", **kw),
    "DualMem-a75":        lambda **kw: _build("DualMem-a75", **kw),
    # Cross-axis ablations: visual-only retrieval × img-only / cap-only injection.
    "DualMem-vis-img":    lambda **kw: _build("DualMem-vis-img", **kw),
    "DualMem-vis-cap":    lambda **kw: _build("DualMem-vis-cap", **kw),
    "CLIPVision":   lambda **kw: _build("CLIPVision", **kw),
    "HYMEM":        lambda **kw: _build("HYMEM", **kw),
    # External 2025-2026 multimodal-agent-memory baselines (Trial-API adapters
    # wrapped to satisfy the F2 MemorySystem interface). The "-lite" suffix
    # signals an HONEST adaptation: see doc/multimodal_baseline_adaptation.md
    # for what was kept vs dropped, per adapter. MMA carries no "-lite"
    # because the R(m|q) reliability formula is reproduced faithfully.
    "WorldMM-lite": lambda **kw: _ext("WorldMM-lite", **kw),
    "M2A-lite":     lambda **kw: _ext("M2A-lite", **kw),
    "MMA-lite":     lambda **kw: _ext("MMA-lite", **kw),
    # Legacy aliases for in-flight runs / older scripts. New experiments
    # should use the "-lite" names.
    "WorldMM":      lambda **kw: _ext("WorldMM-lite", **kw),
    "M2A":          lambda **kw: _ext("M2A-lite", **kw),
    "MMA":          lambda **kw: _ext("MMA-lite", **kw),
}


def make_system(name: str, **kwargs) -> MemorySystem:
    if name not in SYSTEM_REGISTRY:
        raise KeyError(f"Unknown system {name!r}. Known: {list(SYSTEM_REGISTRY)}")
    return SYSTEM_REGISTRY[name](**kwargs)
