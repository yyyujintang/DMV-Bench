"""Injection layer — formats memory entries into VLM-consumable prompts.

An Injector takes a retrieved MemoryEntry and produces a dict with:
    - "text"   : the textual portion that gets prepended to the VLM prompt
    - "images" : the list of image file paths to attach (may be empty)

This is the layer that an injection-cap ceiling would bypass (replace
with the strongest possible "image_only" formatter), to attribute losses
to retrieval vs injection separately.
"""

from dualmem.injection.base import Injector, InjectionPayload
from dualmem.injection.text_only_inject import TextOnlyInject
from dualmem.injection.caption_only_inject import CaptionOnlyInject
from dualmem.injection.image_only_inject import ImageOnlyInject
from dualmem.injection.image_text_inject import ImageTextInject
from dualmem.injection.registry import make_injector, INJECTOR_REGISTRY

__all__ = [
    "Injector", "InjectionPayload",
    "TextOnlyInject", "CaptionOnlyInject", "ImageOnlyInject", "ImageTextInject",
    "make_injector", "INJECTOR_REGISTRY",
]
