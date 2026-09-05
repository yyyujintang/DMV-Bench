"""DMV-Bench: diagnosing long-horizon visual memory in multimodal agents.

The benchmark runs a chain of autonomous shopping sessions over a controlled
1,000-variant storefront. Every product image carries a unique pre-rendered
incidental cue that appears in no text channel, so only a memory architecture
that retains pixels can answer the recall probes.

Package layout:
    agent/       the ReAct encoding agent, the recall-probe runner and the
                 shared prompt surface every system is driven through
    memory/      memory banks (visual, verbal, dual)
    encoders/    visual (CLIP, SigLIP-2, DINOv2/v3) and verbal (SBERT) encoders
    retrieval/   retrievers and the hybrid score fusion
    injection/   what a retrieved entry re-presents to the VLM
    systems/     the named architectures, composed from the four stages above
    baselines/   adapters for external published memory systems
    vlm/         VLM back-end clients
"""
