"""Summarizer evaluation: ROUGE-L + LLM faithfulness judge.

Usage:
    from eval.summarizer_eval import evaluate_summarizer, evaluate_live

    # From pre-generated (messages, summary) pairs:
    report = await evaluate_summarizer(llm, [(messages, summary_text), ...])

    # From live ChunkSummarizer on a conversation:
    report = await evaluate_live(llm, conversation_messages, chunk_size=4, window_size=2)
"""
import asyncio
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False


@dataclass
class SummarizerReport:
    rouge_l_mean: float | None   # None if rouge-score package is not installed
    faithfulness_mean: float     # 1-5 LLM judge score, averaged across samples
    samples: list[dict]          # per-sample: rouge_l, faithfulness, lengths


async def evaluate_summarizer(
    llm: BaseChatModel,
    samples: list[tuple[list[BaseMessage], str]],
) -> SummarizerReport:
    """Evaluate a list of (source_messages, generated_summary) pairs.

    Args:
        llm: Used for the faithfulness judge.
        samples: Each element is (source messages, generated summary text).
    """
    scorer = _rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True) if _HAS_ROUGE else None
    results = []

    for messages, summary in samples:
        source = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
            if isinstance(m.content, str)
        )

        rouge_l = None
        if scorer:
            score = scorer.score(source, summary)
            rouge_l = round(score["rougeL"].fmeasure, 4)

        faithfulness = await _judge_faithfulness(llm, source, summary)
        results.append({
            "rouge_l": rouge_l,
            "faithfulness": faithfulness,
            "summary_chars": len(summary),
            "source_chars": len(source),
        })

    rouge_scores = [r["rouge_l"] for r in results if r["rouge_l"] is not None]
    faith_scores = [r["faithfulness"] for r in results]

    return SummarizerReport(
        rouge_l_mean=round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else None,
        faithfulness_mean=round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0.0,
        samples=results,
    )


async def evaluate_live(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    chunk_size: int = 6,
    window_size: int = 6,
) -> SummarizerReport:
    """Run ChunkSummarizer on messages, then evaluate the produced chunk summaries.

    Args:
        messages: Full conversation history.
        chunk_size: Messages per chunk (must match ChunkSummarizer config).
        window_size: Sliding window size (determines how many messages fall outside).
    """
    from chat_history.summarizer import ChunkSummarizer

    summarizer = ChunkSummarizer(llm=llm, chunk_size=chunk_size)
    await summarizer.update("_eval_live", messages, window_size)
    chunks: list[str] = summarizer._chunks.get("_eval_live", [])

    if not chunks:
        return SummarizerReport(rouge_l_mean=None, faithfulness_mean=0.0, samples=[])

    samples = [
        (messages[i * chunk_size: (i + 1) * chunk_size], summary)
        for i, summary in enumerate(chunks)
    ]
    return await evaluate_summarizer(llm, samples)


async def _judge_faithfulness(llm: BaseChatModel, source: str, summary: str) -> float:
    prompt = f"""You are evaluating a conversation summary for faithfulness.

Source conversation:
{source}

Summary:
{summary}

Rate how faithfully the summary represents the source conversation on a scale of 1 to 5:
1 = Major facts missing or factually wrong
2 = Some important information lost or distorted
3 = Mostly accurate, minor details missing
4 = Accurate and covers the key points
5 = Highly accurate and complete

Respond with ONLY a single integer (1-5)."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip()[0])
    except (ValueError, IndexError):
        return 3.0
