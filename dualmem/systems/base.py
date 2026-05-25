"""MemorySystem — generic composition of (bank, retriever, injector)."""

from __future__ import annotations

from typing import Optional

from dualmem.injection.base import Injector, InjectionPayload
from dualmem.memory.base import MemoryBank
from dualmem.memory.entry import MemoryEntry
from dualmem.retrieval.base import Retriever, RetrievalQuery
from dualmem.types import Trial


class MemorySystem:
    """Composite memory system.

    Constructor takes the three layer-pieces. The caption_fn is an
    optional callable `(image_path) -> str` that the system calls at
    encode time when the bank needs a caption (Caption / DualChannel /
    HYMEM). For systems that don't need captions (TextOnly, RawImage,
    CoMEM-image-only), pass None.

    `oracle_injector` is the injector used for the oracle_retrieval
    ceiling. By default it's the same as `injector`, but some baselines
    (e.g., TextOnly with text_only inject) want a uniform comparison
    that always injects in the SAME format the system natively uses.
    """
    def __init__(
        self,
        name: str,
        bank: MemoryBank,
        retriever: Retriever,
        injector: Injector,
        caption_fn: Optional[callable] = None,
        oracle_injector: Optional[Injector] = None,
    ):
        self.name = name
        self.bank = bank
        self.retriever = retriever
        self.injector = injector
        self.caption_fn = caption_fn
        self.oracle_injector = oracle_injector or injector

    # ---- pipeline operations the ceiling runners call ----

    def reset(self):
        self.bank.reset()

    def encode(self, trial: Trial) -> MemoryEntry:
        caption = None
        if self.caption_fn is not None:
            caption = self.caption_fn(trial.anchor_path)
        return self.bank.encode(
            image_path=trial.anchor_path,
            slug=trial.anchor_slug,
            encode_text=trial.encode_text,
            caption=caption,
        )

    def encode_external(self, image_path: str, slug: str,
                        encode_text: str = "", caption: Optional[str] = None) -> MemoryEntry:
        """Used by ceiling runners to seed distractor memories without
        materialising full Trials."""
        return self.bank.encode(
            image_path=image_path, slug=slug,
            encode_text=encode_text, caption=caption,
        )

    def retrieve(self, query: RetrievalQuery) -> Optional[MemoryEntry]:
        hits = self.retriever.retrieve(query, self.bank.entries(), k=1)
        return hits[0] if hits else None

    def inject(self, entry: MemoryEntry) -> InjectionPayload:
        return self.injector.render(entry)

    def oracle_inject(self, trial: Trial) -> InjectionPayload:
        """Inject the anchor variant directly (skip retrieval entirely).
        Used by the oracle_retrieval ceiling.
        """
        # Find the anchor entry in the bank if present; otherwise synthesise.
        e = self.bank.find_by_slug(trial.anchor_slug)
        if e is None:
            # Synthesize an entry on the fly so injectors that read caption
            # (Caption / DualChannel / HYMEM) have something to render.
            cap = self.caption_fn(trial.anchor_path) if self.caption_fn else None
            e = MemoryEntry(
                slug=trial.anchor_slug,
                image_path=trial.anchor_path,
                encode_text=trial.encode_text,
                caption=cap,
            )
        return self.oracle_injector.render(e)
