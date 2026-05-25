# DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection

Code release accompanying the paper *DMV-Bench: Diagnosing Long-Horizon
Multimodal Agents' Visual Memory with Incidental Cue Injection*. This
repository contains:

- **DMV-Bench** — an interactive, multi-session benchmark for *visual* agent
  memory. The agent runs a chain of $J \in \{5,10,15,50\}$ autonomous
  shopping sessions over a controlled $1{,}000$-variant home-furnishing
  storefront. Every visited product image carries a unique, pre-rendered
  *incidental cue*; a strict L2-leakage contract keeps the cue out of every
  text channel, so only a memory architecture that retains pixels can answer
  the recall probes. Evaluation sweeps *recall reach* $r$ — the
  session-distance between visit and probe — to read off how long an
  incidentally-seen cue survives.

- **DualMem** — a dual-coding-inspired memory architecture that holds a
  visual and a verbal code for every observation, stores both in one bank,
  and fuses the visual (SigLIP-2) and verbal (SBERT) scores at retrieval
  before injecting the top image *and* caption into the next-action VLM.
  DualMem leads every baseline at every chain length on both back-ends
  (Gemini 2.5 Flash and Qwen2.5-VL-7B), with the lead surviving controls
  for memory-bank size and encoding-position bias.

## Repository layout

```
DMV-Bench/
├── dualmem/             core memory architectures (DualMem + baselines)
│   ├── agent/          ReAct shopping agent + multi-session runner
│   ├── baselines/      WorldMM, M2A, MMA, Caption, NoMemory, ContextOnly
│   ├── encoders/       SigLIP-2, SBERT, CLIP, DINOv2/v3 encoders
│   ├── memory/         DualBank, VisualBank, VerbalBank
│   ├── retrieval/      HybridNormRetriever, TextRetriever, VisualRetriever
│   ├── injection/      ImageTextInject, ImageOnlyInject, CaptionOnlyInject
│   ├── inventory/      product catalogue interface
│   ├── systems/        registry + external-baseline adapters
│   ├── vlm/            back-end adapters (Gemini, Qwen-VL via vLLM, etc.)
│   └── metrics_f2.py   per-reach SR aggregator
├── tasks/
│   ├── schema/         RolloutTreeTask + SessionSpec Pydantic models
│   ├── generators/
│   │   └── f2_online_ic.py   incidental-cue task generator (Family 2)
│   ├── scoring/        URL-match scorer
│   └── validators/
├── scripts/
│   └── run_dmvbench_f2.py    paper entry point (J=5/10/15/MC J=50)
├── env/
│   └── frontend/             Next.js storefront serving the catalogue
│       ├── app/, components/, lib/    UI + L2-leakage contract
│       └── prisma/           SQLite schema
├── pipeline/                 image-generation pipeline (Gemini Imagen +
│                             NanoBanana editor) to regenerate the catalogue
├── requirements.txt
└── LICENSE                   Apache-2.0
```

## Getting started

1. **Python deps** — `pip install -r requirements.txt`
2. **Frontend** — `cd env/frontend && npm install && npx prisma migrate dev`
3. **Catalogue images** — the $1{,}000$-variant catalogue images
   (`env/frontend/public/images_v2/`) are too large to ship with code.
   Regenerate them with:

   ```bash
   cd pipeline
   python generate.py --target-count 1000 --backend gemini-imagen
   # then run the cue-editing pass (NanoBanana edits):
   python generate.py --mode edit-cues
   ```

   This populates `env/frontend/public/images_v2/{base,with_cue}/<category>/<style>/`.

4. **Seed the storefront DB** —

   ```bash
   cd env/frontend
   npx tsx scripts/seed_from_pipeline.ts
   ```

5. **Run the dev server** — `npm run dev -- -p 3000`

6. **Run an experiment** —

   ```bash
   # Incidental-Cue task, J=5 chain, Gemini back-end, single seed:
   python scripts/run_dmvbench_f2.py \
       --n-sessions 5 --max-steps 20 \
       --vlm gemini-2.5-flash --base-url http://localhost:3000 \
       --systems DualMem-a75 --seeds 0 \
       --out-dir results/J5_DualMem-a75/
   ```

   Switch to Qwen2.5-VL-7B via vLLM with
   `--vlm qwen-vl-7b-vllm --base-url <vllm_url>`.

## Memory architectures audited

The seven systems share an `encode` / `retrieve` / `inject` interface
(`dualmem/systems/`); DualMem and the four external multimodal baselines all
operate under the same harness, so any per-cell gap reflects the memory
architecture rather than harness drift.

| System              | Encode               | Retrieve                | Inject       |
|---------------------|----------------------|-------------------------|--------------|
| NoMemory            | (none)               | (none)                  | (none)       |
| ContextOnly         | encode_text          | full dump               | text         |
| Caption             | VLM caption          | SBERT cosine            | text         |
| WorldMM-lite        | ep+sem+vis modules   | adaptive iterative      | retrieved    |
| MMA-lite            | semantic store       | reliability-weighted    | text         |
| M2A-lite            | raw + semantic       | dense+BM25+visual fuse  | text         |
| **DualMem (ours)**  | image + caption      | hybrid (SigLIP + SBERT) | image+caption|

DualMem variants (`DualMem-a25`, `DualMem-a75`, `DualMem-vis`, `DualMem-verb`,
`DualMem-img`, `DualMem-cap`, `DualMem-vis-img`, `DualMem-vis-cap`) ablate
the hybrid retrieval-weight α and the injection modality. The headline number
in the paper uses `DualMem-a75` (visual-dominant fusion); `DualMem` (without
an α suffix) defaults to the symmetric α=0.5 baseline.

## Task family

DMV-Bench is built around a *single* task type — **Incidental Cue (IC)** —
instantiated as a chain of $J$ short shopping sessions. In each session a
memoryless ReAct agent browses for ~25 steps and visits ~12 products;
cues are present in product images but are never mentioned in text. After
the chain finishes, recall probes are issued at controlled reach $r$ (the
session-distance between the visit and the probe), and the agent must
emit `navigate(<product-url>)` to the cued product. The probe is scored
by exact URL match against a bijective ground truth.

Long chains are evaluated over a **shared-prefix rollout tree** to amortise
the cost of the early sessions: the first session is run once and $B$
children branch from its memory snapshot, each branching $B$ ways in turn,
to depth $J$. A tree of depth $J$ and branching factor $B$ costs
$(B^J - 1)/(B-1)$ session runs but yields $\sim B^{J-1}$ root-to-leaf
recall paths — roughly a $J\times$ saving over flat re-runs at $B{=}5$.

## Data products

This repo ships **code only**. Two artefact families are regenerable:

- **Catalogue images** (~1.5 GB): rebuild with `pipeline/generate.py`.
- **Family 2 task spines** (`tasks/pool_v2/f2_trees/*.json`): regenerate
  with `tasks/generators/f2_online_ic.py` (the runner does this
  automatically when `--seeds N` is passed).

Per-trial logs and trajectory caches are written under `results/` at run
time and are not included.

## License

Apache-2.0 (see `LICENSE`).
