"""Injector factory."""

from __future__ import annotations

from typing import Any, Callable, Dict

from dualmem.injection.text_only_inject import TextOnlyInject
from dualmem.injection.caption_only_inject import CaptionOnlyInject
from dualmem.injection.image_only_inject import ImageOnlyInject
from dualmem.injection.image_text_inject import ImageTextInject


INJECTOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    "text_only":    TextOnlyInject,
    "caption_only": CaptionOnlyInject,
    "image_only":   ImageOnlyInject,
    "image_text":   ImageTextInject,
}


def make_injector(name: str, **kwargs):
    if name not in INJECTOR_REGISTRY:
        raise KeyError(f"Unknown injector {name!r}. Known: {list(INJECTOR_REGISTRY)}")
    return INJECTOR_REGISTRY[name](**kwargs)
