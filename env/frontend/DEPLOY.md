# DMV-Bench Annotation Tool — Deploy to Vercel + Supabase (no GitHub)

Self-host the annotation site as a public Prolific-accessible URL without
ever pushing source code to GitHub. The Vercel CLI uploads the build
directly from your laptop to Vercel's build infrastructure; no public
repo, no fork, no leak.

---

## Prerequisites

- Vercel account (free tier covers this study comfortably)
- Supabase account (free tier: 500 MB DB, 2 GB egress / month)
- A laptop with Node 18+ installed

---

## 1. Supabase — create the Postgres DB

1. https://supabase.com/dashboard → **New project**
   - Name: `dmv-bench` (or any name)
   - Database password: generate + save (you'll need it for the connection string)
   - Region: pick the one closest to your annotators (US-East / EU-West are common)
   - Plan: Free

2. After ~1 min provisioning, go to **Project Settings → Database → Connection string → URI**.
   You'll see something like:
   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
   ```
   Save this — it's your `DATABASE_URL`.

   > **Important**: Supabase exposes two connection modes — the standard one
   > (port 5432) and the **pgbouncer/connection-pooler** one (port 6543).
   > Vercel's serverless functions create many short-lived connections, so
   > **use the pooler URL for Vercel runtime**:
   > `postgresql://postgres.[ref]:[pwd]@aws-0-[region].pooler.supabase.com:6543/postgres?pgbouncer=true`
   >
   > Use the direct 5432 URL only for one-off migrations (Prisma needs a real
   > connection, not a pooled one, for DDL statements).

3. Optional: in **Database → Roles**, restrict the `postgres` role's network
   access list to your laptop + Vercel egress IPs. For first deploy you can
   leave it open and tighten later.

---

## 2. Local migrate + seed

`prisma/schema.prisma` is kept at `provider = "sqlite"` by default so
`npm run dev` works with no extra setup. Before running migrations against
Supabase, **swap the provider to postgresql** via the helper script:

```bash
npm run prisma:postgres        # rewrites schema.prisma: sqlite → postgresql
```

Then in `env/frontend/`:

```bash
# DIRECT (port 5432) Supabase URL — only for migration.
export DATABASE_URL='postgresql://postgres:[pwd]@db.xxxxx.supabase.co:5432/postgres'

# Generate the Postgres-dialect init migration. Prisma sees the empty
# migrations/ dir and creates one fresh from schema.prisma.
npx prisma migrate dev --name init --schema prisma/schema.prisma

# Regenerate the Prisma client against Postgres
npx prisma generate --schema prisma/schema.prisma

# Seed the catalog (categories, products, variants, images)
npm run seed
```

Sanity check via Supabase **Table Editor** — you should see `Category`,
`Product`, `ProductVariant`, `Annotation4afc`, `AnnotationMechanism`,
`AnnotationWorker` populated/created.

After migrate + seed, flip the schema back so you can resume local dev
with sqlite:

```bash
npm run prisma:sqlite          # rewrites schema.prisma: postgresql → sqlite
npx prisma generate --schema prisma/schema.prisma   # regenerate sqlite client
```

The Vercel build (next section) takes care of the swap automatically on
its own build machine — your local file stays on sqlite.

The legacy SQLite migrations are kept in `prisma/migrations.sqlite-legacy/`
for reference only. Don't try to apply them to Postgres.

---

## 3. Install Vercel CLI + log in

```bash
npm i -g vercel              # global install, one-time
vercel login                 # browser-based auth flow
```

Sign in with your Vercel account (email / GitHub OAuth is fine — OAuth
just authenticates *you*, no repository is created or read).

---

## 4. First-time link

From `env/frontend/`:

```bash
vercel
```

The CLI walks you through:

```
? Set up and deploy "~/.../env/frontend"? [Y/n] y
? Which scope do you want to deploy to?       <your-account>
? Link to existing project?                    n        ← first time only
? What's your project's name?                  dmv-bench
? In which directory is your code located?     ./       ← we're already in env/frontend
```

It detects Next.js automatically and runs a **preview** build first. You'll get
a URL like `https://dmv-bench-abc123-yourname.vercel.app` — this is a preview
deploy, **not yet wired to env vars**, so it'll fail the DB check. That's
expected.

The CLI writes a `.vercel/` directory locally with the project link metadata.
Don't commit `.vercel/` if you ever do version-control this code — it
contains your project ID and org ID (already covered by Vercel's default
`.gitignore` template, but worth knowing).

### 4a. Override the build command

The local schema is sqlite (for `npm run dev`), but Vercel needs to build
against postgresql. Set the **Build Command** override so Vercel runs the
swap-then-build wrapper instead of plain `next build`:

```bash
# Via CLI — once, after the first link:
vercel project ls                                # confirm the project name
# Then open the dashboard for that project → Settings → Build & Development
#   Build Command:    npm run build:vercel
#   Install Command:  npm install         (default is fine)
#   Output Directory: .next               (default is fine)
```

The `build:vercel` npm script runs `scripts/build-vercel.mjs` (which calls
`scripts/swap-provider.mjs postgresql` to rewrite `prisma/schema.prisma`),
then `prisma generate`, then `next build`. The mutation happens on
Vercel's ephemeral build VM only.

---

## 5. Set the environment variable

```bash
# Add DATABASE_URL = the POOLER url (6543) to production, preview, and dev
vercel env add DATABASE_URL production
# paste:  postgresql://postgres.[ref]:[pwd]@aws-0-[region].pooler.supabase.com:6543/postgres?pgbouncer=true

vercel env add DATABASE_URL preview
vercel env add DATABASE_URL development        # only if you want to pull it locally with `vercel env pull`
```

Or set it once in the **Vercel dashboard**:
`vercel.com/dashboard → your project → Settings → Environment Variables`.
You can paste the same value for all three environments.

---

## 6. Production deploy

```bash
vercel --prod
```

That uploads the current directory (respecting `.gitignore` + `.vercelignore`
if present), builds on Vercel's infrastructure, and assigns the production
URL. Output looks like:

```
✅  Production: https://dmv-bench.vercel.app
```

Every subsequent code change is one `vercel --prod` away. No git push,
no automatic deploy webhook — **you** decide when a new version goes live.

---

## 7. Configure Prolific study

In your Prolific study setup, set the **study URL** to:

```
https://dmv-bench.vercel.app/annotate?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

Prolific substitutes the curly-brace tokens at runtime so each worker arrives
with their unique IDs.

Worker flow:

1. Click Prolific link → middleware redirects through `/api/annotate/entry`
   → cookie set → consent page
2. Consent + practice → task selector
3. Pick task 1 / 2 / 3 in any order
4. Click **Finish** → `/annotate/done` shows the completion code
   (format `DMV-XXXX-XXXX`)
5. Worker pastes the code into Prolific to claim payment

Prolific's **completion code field** should match the prefix `DMV-` so the
validator on their side recognises it.

---

## 8. Smoke-test the deploy

From your laptop:

```
https://dmv-bench.vercel.app/annotate?PROLIFIC_PID=test_smoke_1&STUDY_ID=test_study&SESSION_ID=test_session
```

Walk through: consent → practice → at least one task → Finish → done.
Inspect the `AnnotationWorker` row in Supabase to confirm `completionCode`
was saved.

Clean up smoke-test rows:

```sql
delete from "AnnotationWorker" where "prolificId" like 'test_%';
```

---

## 9. Updating the deploy

Code change?

```bash
vercel --prod               # rebuild + redeploy to the same production URL
```

Schema change?

```bash
# Run migrations from your laptop against the direct (5432) URL, then redeploy
export DATABASE_URL='postgresql://...:5432/postgres'
npx prisma migrate dev --name <change-name> --schema prisma/schema.prisma
vercel --prod                                    # picks up new Prisma client at build time
```

To deploy a non-production preview (e.g., to share a draft with a
collaborator):

```bash
vercel                      # creates a temporary <hash>.vercel.app preview URL
```

---

## 10. Operational notes

- **Concurrency**: the Supabase free tier handles ~50 concurrent workers
  without issue (Prisma + pgbouncer covers connection limits).
- **Logs**: `vercel logs` streams live function logs; the dashboard
  **Functions** tab has per-route history. Supabase **Logs** shows DB
  activity.
- **Cost guardrails**: Vercel free tier counts function invocations + edge
  middleware; the annotation flow is bounded (a few dozen requests per
  worker), so 200 workers ≈ ~6k invocations — comfortably under the 100k/mo
  hobby limit.
- **Backups**: Supabase takes daily backups on the free tier. For the paper
  run, export the four annotation tables to CSV at the end via the SQL editor:
  ```sql
  copy (select * from "Annotation4afc") to stdout with csv header;
  copy (select * from "AnnotationMechanism") to stdout with csv header;
  copy (select * from "AnnotationWorker") to stdout with csv header;
  ```

---

## 11. Source-secrecy posture

What lives where:

| Surface | What it sees | Public? |
|---|---|---|
| Your laptop | Full source, full DB credentials | Private (you) |
| Vercel build infra | Full source (during build) + DATABASE_URL env var | Internal to Vercel, not exposed externally; deploys default to "private to your account" |
| Vercel-served URL | Compiled JS bundles + server handlers — same code an annotator's browser fetches | Public, but is the **compiled** app, not the source |
| Supabase | Database rows + connection metadata | Internal to Supabase, restricted to your project |
| Prolific worker | Whatever a normal browser sees on `/annotate/*` | Public surface (intentional) |

**No GitHub, GitLab, or other VCS hosting is involved.** The source never
leaves your laptop except when piped into a Vercel build.

If you want a stronger isolation guarantee, gate the Vercel project with
**Password Protection** (Settings → Deployment Protection on the dashboard
— Pro feature, $20/month) so even the public URL requires a shared password
to view. You'd then bake the password into the Prolific link as a query
string, or share it in the study instructions.

---

## 12. Security hardening (optional, after first pilot)

These weren't critical for a pilot but worth adding for the main run:

- **Rate limit by IP** (Upstash Redis + middleware) — caps a single bad actor
  to N entry hits / hour
- **Custom domain** — Vercel free supports `your-domain.com`; nicer-looking
  Prolific URL
- **Geo restrictions** — Vercel firewall can deny non-Prolific-target regions
- **Worker-row TTL** — periodic cleanup of abandoned sessions older than
  N days to keep the DB small

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `prisma migrate dev` hangs on Supabase | You're using the pooler URL (6543) — switch to direct (5432) for migrations |
| `vercel` upload includes `node_modules` and times out | Add `node_modules` to `.vercelignore` (the default `.gitignore` is honored automatically, but worth double-checking) |
| Worker creates row but `/annotate` shows "Access via Prolific only" | Host-header mismatch — see `/api/annotate/entry/route.ts` `backToAnnotate()`; verify Vercel isn't rewriting Host on the way back |
| Vercel build fails on `prisma generate` | The `postinstall` script in `package.json` should run it; if you stripped that, add `"postinstall": "prisma generate --schema prisma/schema.prisma"` back |
| Vercel build fails: image too large | The `/public/images/` directory may exceed the 100 MB deployment limit. Move large images to Supabase Storage and rewrite `primaryImage` URLs |
| Completion code shows up but Prolific marks invalid | Confirm the code prefix in the Prolific study matches what your worker pasted; the codes are stored in `AnnotationWorker.completionCode` for audit |
| `vercel --prod` errors with "Project not linked" | Re-run `vercel` (without `--prod`) to relink, or delete `.vercel/` and start over |
