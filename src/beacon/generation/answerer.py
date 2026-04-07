"""
Answerer — generates grounded, cited answers using Claude and retrieved context.

This is the core RAG generation step. Given a user question and retrieved chunks,
it constructs a prompt that instructs Claude to:
  1. Answer ONLY from the provided context (grounding)
  2. Cite specific sources for every claim (traceability)
  3. Say "I don't have enough information" when context is insufficient (honesty)

The citation format uses [Source: filename] inline tags so the user can
verify every claim against the original document.
"""

from typing import List, Dict, Optional

import anthropic

from beacon.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, TEMPERATURE


SYSTEM_PROMPT = """You are Beacon, a knowledge assistant that answers questions using ONLY the provided context documents.

RULES — follow these strictly:
1. ONLY use information from the provided context to answer. Never use your general knowledge.
2. CITE every claim with [Source: filename] tags inline.
3. If the context does not contain enough information to answer, say: "I don't have enough information in the loaded documents to answer this question."
4. If different sources provide conflicting information, note the discrepancy and cite both.
5. Be concise and direct. Answer the question, cite the sources, and stop.
6. Use the same language as the question.
7. When listing information, organise it clearly but don't add information beyond what's in the context.

CITATION FORMAT:
- Inline citations: "The return window is 30 days [Source: return-policy.pdf]"
- Multiple sources: "This applies to all regions [Source: policy.pdf] [Source: faq.md]"
"""


def build_context_block(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not chunks:
        return "No relevant documents found."

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"--- Source: {chunk['source']} (relevance: {chunk['score']}) ---\n"
            f"{chunk['text']}\n"
        )

    return "\n".join(blocks)


def generate_answer(
    question: str,
    chunks: List[Dict],
    conversation_history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate a grounded, cited answer using Claude.

    Args:
        question: The user's question
        chunks: Retrieved context chunks from the vector store
        conversation_history: Optional prior messages for multi-turn context

    Returns:
        Dict with 'answer', 'sources_used', and 'chunks_provided'
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the context
    context_block = build_context_block(chunks)

    # Build the user message with context + question
    user_message = (
        f"CONTEXT DOCUMENTS:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using ONLY the context above. Cite every claim with [Source: filename]."
    )

    # Build message history
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Call Claude
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text

    # Extract unique sources referenced in the answer
    import re
    source_pattern = r"\[Source:\s*(.+?)\]"
    sources_cited = sorted(set(re.findall(source_pattern, answer)))

    return {
        "answer": answer,
        "sources_cited": sources_cited,
        "chunks_provided": len(chunks),
        "model": CLAUDE_MODEL,
        "tokens_used": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    }
