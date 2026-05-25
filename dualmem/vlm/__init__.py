"""VLM clients for VisMem-Diag.

Uniform `VLMClient` protocol — every backbone exposes:
  - `four_afc(anchor, candidates, instructions)` returning `VLMResponse`
  - `generate_freeform(system, user, primary_image=, extra_images=)` for the
    Playwright agent loop

Concrete backends (closed-source APIs):
  - GeminiClient        — Gemini 2.5 Pro / Flash (google-genai)
  - OpenAIClient        — GPT-4o family (openai)
  - AnthropicClient     — Claude 3.7 Sonnet (anthropic)

Concrete backends (open-source, local):
  - QwenVLClient        — Qwen2.5-VL 3B / 7B Instruct (transformers)

Plus:
  - StubClient          — deterministic test double (no API, no GPU)

See `doc/vlm_backends.md` for the backbone-coverage policy. Main table
reports the best vision-memory method on 1 open backbone + 3 closed
backbones (Qwen2.5-VL-7B / Gemini-2.5-Pro / GPT-4o / Claude-3.7-Sonnet);
ablation runs go through the open backbones (Qwen2.5-VL 3B + 7B) at scale.
"""

from dualmem.vlm.base import VLMClient, VLMResponse
from dualmem.vlm.gemini import GeminiClient
from dualmem.vlm.openai_client import OpenAIClient
from dualmem.vlm.anthropic_client import AnthropicClient
from dualmem.vlm.qwen_vl import QwenVLClient
from dualmem.vlm.stub import StubClient

__all__ = [
    "VLMClient",
    "VLMResponse",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
    "QwenVLClient",
    "StubClient",
    "make_vlm",
    "BACKEND_DEFAULTS",
]


# Canonical default model id per backbone family. Aliases like "gemini" or
# "claude" resolve here; pass `model=` to override.
BACKEND_DEFAULTS = {
    # Closed-source APIs
    "gemini":         "gemini-2.5-pro",
    "gemini-pro":     "gemini-2.5-pro",
    "gemini-flash":   "gemini-2.5-flash",
    "openai":         "gpt-4o",
    "gpt-4o":         "gpt-4o",
    "gpt-4o-mini":    "gpt-4o-mini",
    "anthropic":      "claude-3-7-sonnet-20250219",
    "claude":         "claude-3-7-sonnet-20250219",
    "claude-3.7":     "claude-3-7-sonnet-20250219",
    # Open-source local
    "qwen-vl":        "qwen-vl-7b",
    "qwen-vl-3b":     "qwen-vl-3b",
    "qwen-vl-7b":     "qwen-vl-7b",
    "qwen2.5-vl-3b":  "qwen-vl-3b",
    "qwen2.5-vl-7b":  "qwen-vl-7b",
    # Stub
    "stub":           "stub-v0",
}


def make_vlm(backend: str, **kwargs):
    """Factory.

    `backend` is any of the keys in BACKEND_DEFAULTS, or a fully-qualified
    HF repo id for Qwen-VL (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").

    Pass `model=` to override the default model id for a family. Other
    kwargs are forwarded to the client constructor (e.g. `device=`,
    `max_pixels=` for QwenVLClient; `max_retries=` for the API clients).
    """
    name = backend.lower().strip()

    # ---- Closed-source APIs --------------------------------------------------
    if name.startswith("gemini"):
        model = kwargs.pop("model", BACKEND_DEFAULTS.get(name, name if "-" in name else "gemini-2.5-pro"))
        return GeminiClient(model=model, **kwargs)

    if name.startswith("openai") or name.startswith("gpt"):
        model = kwargs.pop("model", BACKEND_DEFAULTS.get(name, "gpt-4o"))
        return OpenAIClient(model=model, **kwargs)

    if name.startswith("anthropic") or name.startswith("claude"):
        model = kwargs.pop("model", BACKEND_DEFAULTS.get(name, "claude-3-7-sonnet-20250219"))
        return AnthropicClient(model=model, **kwargs)

    # ---- Open-source via vLLM (OpenAI-compatible HTTP) -----------------------
    # `qwen-vllm` / `qwen-vl-7b-vllm` / `qwen2.5-vl-7b-vllm` → use the
    # locally-served vLLM endpoint (set `VLLM_URL` env, default
    # http://localhost:8000/v1). Massively higher throughput than the
    # transformers-based QwenVLClient because vLLM does continuous batching
    # across all concurrent workers.
    if name.endswith("-vllm") or name.startswith("vllm"):
        import os as _os
        repo = ("Qwen/Qwen2.5-VL-7B-Instruct" if "7b" in name or name == "qwen-vllm"
                else "Qwen/Qwen2.5-VL-3B-Instruct" if "3b" in name
                else "Qwen/Qwen2.5-VL-7B-Instruct")
        model = kwargs.pop("model", repo)
        base_url = kwargs.pop("base_url", _os.environ.get(
            "VLLM_URL", "http://localhost:8000/v1"))
        return OpenAIClient(model=model, base_url=base_url, **kwargs)

    # ---- Open-source local (transformers, single-process; smoke / fallback) --
    if name.startswith("qwen") or "/" in backend:  # `/` means an HF repo id was passed
        alias = BACKEND_DEFAULTS.get(name, backend)  # preserves repo-id case
        model = kwargs.pop("model", alias)
        return QwenVLClient(model=model, **kwargs)

    # ---- Test stub -----------------------------------------------------------
    if name == "stub":
        return StubClient(**kwargs)

    raise ValueError(
        f"Unknown VLM backend: {backend!r}. "
        f"Known: {sorted(BACKEND_DEFAULTS)}"
    )
