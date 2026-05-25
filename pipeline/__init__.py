"""Phase A image-pipeline package.

Build-time refactor under Phase A of the v2 → OSS-backend migration.
The Gemini backend is preserved unchanged behaviour-wise; what's new is:

  - Prompts externalised to `prompts/v1/*.yaml` with content hashing
  - `ImageBackend` Protocol so a Qwen/FLUX implementation can drop in
    later without touching the generator driver
  - Per-image manifest at `pipeline/manifest.jsonl`
  - Post-generation QC (CLIP + Gemini vision) with retry-on-fail
  - `incidentalDetails` tagged BEFORE generation and injected into the
    prompt so the post-gen QC has a concrete claim to verify

See IMAGE_PIPELINE_CURRENT_STATE.md §10 and §11 for the migration
context this package implements.
"""
__all__ = ["prompts", "backends", "qc", "manifest", "generate"]
