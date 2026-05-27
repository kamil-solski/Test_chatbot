"""Labeled test data shared across all eval components."""
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from chat_history.router import Intent


@dataclass
class RouterCase:
    id: int
    prompt: str
    expected_intent: Intent
    notes: str = ""


ROUTER_CASES: list[RouterCase] = [
    RouterCase(1, "Who are you?", Intent.CONTINUATION),
    RouterCase(2, "What is prompt engineering?", Intent.NORMAL),
    RouterCase(3, "Is PE important?", Intent.CONTINUATION),
    RouterCase(4, "What other skills?", Intent.CONTINUATION),
    RouterCase(
        5,
        "Is prompt engineering becoming less important as models get smarter, and is context engineering the better framing?",
        Intent.NORMAL,
    ),
    RouterCase(6, "Provide prompt engineering techniques examples.", Intent.NORMAL),
    RouterCase(
        7,
        "Combine the skills you listed when I asked 'What other skills?' with the PE techniques you described into a 3-step roadmap.",
        Intent.SEMANTIC_RECALL,
        "known under-classification: 'you listed', 'I asked', 'you described' not in recall patterns",
    ),
    RouterCase(8, "What was the first question?", Intent.POSITION),
    RouterCase(9, "My favorite language is Python.", Intent.NORMAL),
    RouterCase(
        10,
        "What is my favorite language?",
        Intent.SEMANTIC_RECALL,
        "known under-classification: no recall trigger words",
    ),
    RouterCase(11, "What was the first topic we discussed?", Intent.POSITION),
    RouterCase(
        12,
        "Summarize what we've covered so far.",
        Intent.SEMANTIC_RECALL,
        "known under-classification: \"we've\" contraction bypasses recall regex",
    ),
    RouterCase(
        13,
        "Earlier I told you my favorite language — what did you say it was?",
        Intent.SEMANTIC_RECALL,
    ),
    RouterCase(14, "What was my last question?", Intent.POSITION),
    RouterCase(15, "What was my 3rd question?", Intent.POSITION),
    RouterCase(
        16,
        "Going back to prompt engineering — give me one more technique.",
        Intent.TOPIC_RECALL,
    ),
    RouterCase(17, "Remember when I told you my favorite language?", Intent.SEMANTIC_RECALL),
    RouterCase(18, "Tell me about the weather today.", Intent.NORMAL),
]

# Sample conversation for summarizer and retrieval eval.
# Long enough to produce at least one chunk when chunk_size=4.
SAMPLE_CONVERSATION = [
    HumanMessage(content="What is prompt engineering?"),
    AIMessage(
        content=(
            "Prompt engineering is the practice of designing and refining inputs to language models "
            "to produce better, more accurate, or more useful outputs. Key techniques include "
            "few-shot examples, chain-of-thought prompting, role assignment, and output format "
            "specification."
        )
    ),
    HumanMessage(content="Is it still important with newer models?"),
    AIMessage(
        content=(
            "Yes — even as models improve, prompt design remains critical for reliable structured "
            "outputs, cost control, and task-specific accuracy. Some argue it's shifting toward "
            "'context engineering': curating what goes into the context window rather than phrasing "
            "instructions perfectly."
        )
    ),
    HumanMessage(content="My favorite language is Python."),
    AIMessage(
        content=(
            "Great choice! Python is dominant in AI/ML work — LangChain, PyTorch, FastAPI, "
            "and most LLM SDKs are Python-first."
        )
    ),
    HumanMessage(content="What was the first topic we discussed?"),
    AIMessage(content="The first topic was prompt engineering."),
]

# Reference answers paired to ROUTER_CASES for e2e evaluation.
# Empty string means no reference (judge on coherence + groundedness only).
E2E_REFERENCES: dict[int, str] = {
    1: "I am a helpful AI assistant.",
    2: "Prompt engineering is the practice of designing inputs to LLMs to improve output quality.",
    8: "Your first question was 'What is prompt engineering?'",
    9: "",  # fact injection — no expected answer
    10: "Your favorite language is Python.",
    11: "The first topic we discussed was prompt engineering.",
    14: "Your last question was 'What was the first topic we discussed?'",
    15: "Your 3rd question was about Python being your favorite language.",
    17: "Yes, you mentioned earlier that your favorite language is Python.",
    18: "",  # out-of-scope — no reference needed
}
