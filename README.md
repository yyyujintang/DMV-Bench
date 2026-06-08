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

1. **Python deps** —

   ```bash
   pip install -r requirements.txt
   # one-time: fetch the headless browser the agent drives the storefront with
   playwright install chromium
   ```

   For a GPU box, install the CUDA build of PyTorch first (see the note in
   `requirements.txt`); the encoders fall back to CPU otherwise.

2. **Catalogue images** — every storefront product carries a unique baked-in
   incidental cue, so the runner and the storefront share one `with_cue` image
   set (~1.5 GB). We release the original catalogue as a public Hugging Face
   dataset — [`yyyujintang/DMV-Bench-Images`](https://huggingface.co/datasets/yyyujintang/DMV-Bench-Images).
   Fetch it into the layout the code expects (no token needed):

   ```bash
   python scripts/download_images.py            # add --link-frontend to save ~1.5 GB
   ```

   This populates `data/vismem_diag_v2/images/with_cue/` (read by the runner)
   and `env/frontend/public/images_v2/` (served by the storefront). Point at a
   different mirror with `--repo-id` or `DMVBENCH_IMAGES_REPO`. To instead
   regenerate the catalogue from scratch, see `pipeline/` (requires a Gemini
   Imagen / NanoBanana key).

3. **Frontend** —

   ```bash
   cd env/frontend
   npm install                       # also runs `prisma generate`
   cp .env.example .env              # sets DATABASE_URL=file:../prisma/dev.db
   npx prisma db push                # create the SQLite schema
   npm run seed:v2                   # load the 1,000-variant catalogue
   ```

   > Use `prisma db push` (not `migrate dev`): the committed migrations target
   > Postgres for the hosted-deploy path, while local dev runs on SQLite.

4. **Run the dev server** — `npm run dev -- -p 3000`

5. **Run an experiment** (from the repo root, with a `GEMINI_API_KEY` exported) —

   ```bash
   # Incidental-Cue task, J=5 chain, Gemini back-end, single seed:
   python scripts/run_dmvbench_f2.py \
       --n-sessions 5 --max-steps 20 \
       --vlm gemini-2.5-flash --base-url http://localhost:3000 \
       --systems DualMem-a75 --seeds 0 \
       --out-dir results/J5_DualMem-a75/
   ```

   Writes `f2_summary.csv`, `f2_per_probe.csv`, and per-trial logs under
   `--out-dir`. Switch to Qwen2.5-VL-7B via vLLM with
   `--vlm qwen-vl-7b-vllm --base-url <vllm_url>`.

## Memory architectures audited

The seven systems share an `encode` / `retrieve` / `inject` interface
(`dualmem/systems/`); DualMem and the four external multimodal baselines all
operate under the same harness, so any per-cell gap reflects the memory
architecture rather than harness drift.

| System              | Encode               | Retrieve                | Inject       |
|---------------------|----------------------|-------------------------|--------------|
| NoMemory            | (none)               | (none)                  | (none)       |
| LongContext         | encode_text          | full dump               | text         |
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

This repo ships **code plus the small JSON sources** (catalogue seed,
pricing/naming, cue registry under `env/scripts/` and `data/vismem_diag_v2/`).
Two large artefact families live outside git:

- **Catalogue images** (~1.5 GB, `with_cue` set): released as the public
  Hugging Face dataset
  [`yyyujintang/DMV-Bench-Images`](https://huggingface.co/datasets/yyyujintang/DMV-Bench-Images);
  fetch with `scripts/download_images.py`, or rebuild with `pipeline/generate.py`.
- **Family 2 task spines** (`tasks/pool_v2/f2_trees/*.json`): regenerate
  with `tasks/generators/f2_online_ic.py` (the runner does this
  automatically when `--seeds N` is passed).

Per-trial logs and trajectory caches are written under `results/` at run
time and are not included.

## License

Apache-2.0 (see `LICENSE`).
