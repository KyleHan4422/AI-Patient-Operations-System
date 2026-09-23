"""How retrieved passages are shown to the agent.

Retrieval itself is rag/retrieve.py. This module decides what the model sees of
it, which is a separate decision and a load-bearing one:

  - every passage is labelled `[chunk N]`, and N is the id a citation has to
    name. Without a handle the model cannot cite, and without a citation the
    grounding check has nothing to verify against.
  - the heading path and the effective date travel with the text, so the model
    is never reasoning over a paragraph whose origin it cannot see.
  - "nothing qualified" is spelled out rather than sent as an empty string. A
    model shown nothing invents something; a model told "no passage reached the
    threshold" abstains, which is the behaviour this layer is built for.

Scores are not shown. They are calibrated for retrieval's own decision (rag
returns nothing below the threshold), and a number in the prompt only invites
the model to do its own thresholding on top.
"""

from __future__ import annotations

import textwrap

from patient_ops.rag.retrieve import Retrieval

# Long passages are already bounded by the chunker, so this only catches an
# unusually long section; it keeps one search from eating the context window.
MAX_PASSAGE_CHARS = 1200

NOTHING_FOUND = (
    "No passage in the clinic's documents was close enough to this question. "
    "The documents do not cover it: say so, do not answer from general knowledge."
)


def for_model(found: Retrieval) -> str:
    if not found.chunks:
        return NOTHING_FOUND
    return "\n\n".join(
        f"[chunk {c.chunk_id}] {c.heading_path}\n"
        f"({c.source_path}, as of {c.effective_date})\n"
        f"{textwrap.shorten(c.content, MAX_PASSAGE_CHARS, placeholder=' ...')}"
        for c in found.chunks
    )
