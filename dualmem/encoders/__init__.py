"""Pluggable encoders for DMV-Bench.

A VisualEncoder maps an image (and optionally a query string) into a dense
vector. Encoders are the visual/textual representation backbone of the
retrieval layer. They are deliberately decoupled from the memory bank
(which only knows how to store entries) and from the retriever (which
only knows how to score memories against a query).

Three encoder families:
    - CLIP   : aligned image + text, native text->image retrieval
    - DINOv2 : image-only, very strong dense vision rep, no aligned text encoder
    - DINOv3 : image-only, newest backbone
Plus:
    - SBERT  : text-only, for verbal-channel retrieval
    - Stub   : deterministic hash, used in CI and unit tests

Use `make_encoder(name, ...)` to construct an encoder by name. DINOv2/v3
expose only `embed_image`; CLIP also exposes `embed_text` (and reports
`supports_text_query == True`).
"""

from dualmem.encoders.base import VisualEncoder, EncoderCapability
from dualmem.encoders.registry import make_encoder, ENCODER_REGISTRY

__all__ = ["VisualEncoder", "EncoderCapability", "make_encoder", "ENCODER_REGISTRY"]
