"""Shared agent surface: what the VLM sees, and how memory is rendered into it.

Every memory system under test is driven through exactly these functions -- the
same system prompt, the same observation type, the same retrieval call and the
same prompt layout -- so any difference between systems comes from the memory
architecture and never from the harness.

Both VLM back-ends (Gemini 2.5 Flash and Qwen2.5-VL-7B) receive an identical
system prompt; only the model weights differ.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from dualmem.retrieval.base import RetrievalQuery


# A product detail page: /product/<8-hex urlHash>.
PRODUCT_RE = re.compile(r"^/product/([0-9a-f]{6,16})$")


@dataclass
class Observation:
    """What the agent sees at step t (analogue of a Playwright page snapshot)."""
    url: str
    image_path: str            # disk PNG; "" if no image (category/collection page)
    title: str = ""
    description: str = ""      # the variant's display name + style + category


SYSTEM_PROMPT_TEMPLATE = """You are a shopping-assistant agent in an e-commerce
website. You receive (i) the customer's instruction, (ii) the conversation so
far, (iii) memory context from prior sessions (may be empty), and (iv) the
current page. Decide ONE action.

URL patterns that exist on the site (other patterns return 404):
  /                                  — home
  /category/<slug>                   — category listing, e.g. /category/vase
  /collection/<slug>-<style>         — styled collection, where <style> is
                                        EXACTLY ONE OF: modern, minimalist,
                                        vintage, industrial, scandinavian,
                                        bohemian, mid_century, rustic, japandi,
                                        art_deco
                                        (e.g. /collection/vase-modern,
                                         /collection/chair-rustic)
  /product/<hash>                    — product detail; <hash> is an 8-hex id
  /wishlist /cart                    — shopping pages

Valid category slugs (use EXACTLY these — all singular, no trailing 's'):
  chair, sofa, lamp, cushion, vase, rug, table, bookshelf, plant_pot, wall_art

Action vocabulary (use EXACT syntax — wrong format = no-op):
  navigate("/category/<slug>")           — go to a category listing
  navigate("/collection/<slug>-<style>") — style ∈ {modern, minimalist, vintage, industrial}
  navigate("/product/<hash>")            — go to a product detail page directly
  click_index(N)                         — on a category or collection page,
                                            click the N-th visible product card
                                            (N=0..3, top-left → bottom-right).
                                            PREFERRED over guessing /product/<hash>.
  add_to_wishlist                        — on a product detail page, add it (TERMINAL action,
                                            use it when you've found the right product)
  done                                   — finish if you are stuck (counts as failure)

Some user requests may ask you to navigate to a product the customer has
previously visited. You cannot see earlier sessions in the conversation — only
the [Memory context] block bridges them.

Strategy hints:
  - The [Memory context] block lists products recalled from prior sessions, each
    with its /product/<hash> URL and image. If a remembered product matches what
    the customer is asking for, navigate("/product/<hash>") DIRECTLY — this is
    the fastest and intended path. You DO know the exact hash in that case.
  - Inspect the memory image(s): pick the product whose image matches what the
    customer is describing, then navigate to that product's hash.
  - Only if memory gives you nothing usable: from a category page use
    navigate("/collection/<slug>-<style>") then click_index(N) on the N-th
    product card. You can NOT guess /product/<hash> URLs you have never seen.
  - Once on the /product/<hash> page the customer asked for, emit add_to_wishlist
    — that ends the session successfully.

Reply in ReAct format, ONE Thought line + ONE Action line:
  Thought: <one sentence reasoning>
  Action: <action>
"""


def _format_history(messages: List[dict], max_turns: int = 120) -> str:
    """In-session conversation history — the WHOLE session. Sessions are short
    (a ~10-20-step WebArena-style task), so the agent sees every turn it took
    this session; nothing is windowed away. `max_turns` is only a runaway
    safety cap (keeps the opening task turn + the most recent turns)."""
    if not messages:
        return "(no history yet)"
    if len(messages) > max_turns + 1:
        shown = [messages[0]] + messages[-max_turns:]
        elided = len(messages) - len(shown)
    else:
        shown, elided = messages, 0
    out = []
    for i, m in enumerate(shown):
        prefix = {"user": "User", "agent": "Agent"}.get(m["role"], m["role"])
        out.append(f"  {prefix}: {m['text']}")
        if elided and i == 0:
            out.append(f"  ... ({elided} earlier turns elided) ...")
    return "\n".join(out)


def _format_memory_payload(system, hits) -> Tuple[str, List[str]]:
    """Render up to k retrieved entries through the system's injector, in rank
    order. Returns (text_block, image_paths) where the image_paths line up
    with the 'memory image N' references in the text — so the VLM can pair
    each candidate's URL with its picture.

    Per-baseline modality is preserved: visual injectors attach images, text/
    caption injectors attach none. Every injector now also names the product
    URL (see dualmem.injection.base.entry_url) so the agent can navigate."""
    if not hits:
        return "(no memory retrieved this step)", []
    lines = [
        f"{len(hits)} item(s) recalled from earlier sessions, most relevant "
        f"first. Any attached memory images appear AFTER the current-page "
        f"image, in the order referenced below.",
    ]
    images: List[str] = []
    for i, h in enumerate(hits, 1):
        pl = system.injector.render(h)
        note = ""
        for ip in (pl.images or []):
            if ip:
                images.append(ip)
                note += f" [memory image {len(images)}]"
        body = (pl.text or "(no text)").replace("\n", "\n      ")
        lines.append(f"  ({i}) {body}{note}")
    return "\n".join(lines), images


def _build_prompt(
    subtask_description: str,
    history_messages: List[dict],
    memory_text: str,
    current_obs: Observation,
    step: int,
    cap: int,
) -> str:
    return (
        f"Step {step}/{cap}.\n\n"
        f"[Subtask]\n  {subtask_description}\n\n"
        f"[Conversation so far]\n{_format_history(history_messages)}\n\n"
        f"[Memory context retrieved this step]\n  {memory_text}\n\n"
        f"[Current page]\n  URL: {current_obs.url}\n"
        f"  Title: {current_obs.title}\n"
        f"  Description: {current_obs.description}\n\n"
        "Reply with one Thought + one Action."
    )


def _retrieve_step(system, query_text: str, k: int = 5):
    """Per-step retrieve. Returns (hits, top-k slugs). The caller renders each
    hit through the system's injector (see _format_memory_payload) so the
    agent gets all k ranked candidates, not just the top-1.

    Results are de-duplicated by slug (keeping the highest-ranked occurrence)
    so the agent never sees the same product twice in one memory block — we
    over-fetch 2k then trim to k distinct slugs."""
    raw = system.retriever.retrieve(
        RetrievalQuery(recall_text=query_text, anchor_slug=None),
        system.bank.entries(), k=k * 2,
    )
    seen: set = set()
    hits = []
    for h in raw:
        if h.slug in seen:
            continue
        seen.add(h.slug)
        hits.append(h)
        if len(hits) >= k:
            break
    return hits, [h.slug for h in hits]


def _bank_has_slug(system, slug: str) -> bool:
    """True if the bank already holds an entry for this slug — a product is
    one memory; re-observing it must not create a duplicate bank entry."""
    return any(getattr(e, "slug", None) == slug for e in system.bank.entries())
