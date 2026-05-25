"""Encoder factory.

Usage:
    enc = make_encoder("clip")
    enc = make_encoder("dinov2", eager=True)        # CPU inference path
    enc = make_encoder("sbert")
    enc = make_encoder("stub")
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from dualmem.encoders.clip_encoder import ClipEncoder
from dualmem.encoders.sbert_encoder import SbertEncoder
from dualmem.encoders.dinov2_encoder import Dinov2Encoder
from dualmem.encoders.dinov3_encoder import Dinov3Encoder
from dualmem.encoders.siglip2_encoder import Siglip2Encoder
from dualmem.encoders.stub import StubEncoder


ENCODER_REGISTRY: Dict[str, Callable[..., Any]] = {
    "clip":    ClipEncoder,
    "sbert":   SbertEncoder,
    "dinov2":  Dinov2Encoder,
    "dinov3":  Dinov3Encoder,
    "siglip2": Siglip2Encoder,
    "stub":    StubEncoder,
}


def make_encoder(name: str, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder {name!r}. Known: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](**kwargs)
