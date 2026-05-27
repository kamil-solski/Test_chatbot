"""Retrieval evaluation: context precision and recall via LLM relevance judge.

Context Precision  = relevant_retrieved / total_retrieved
Context Recall     = relevant_retrieved / total_relevant_in_pool

Both metrics use an LLM binary relevance judge — no manual relevance labels required.

Usage:
    from eval.retrieval_eval import evaluate_retrieval, RetrievalReport

    samples = [
        {
            "query": "What is prompt engineering?",
            "retrieved": [doc1, doc2],        # list[Document] from store.retrieve()
            "all_docs": [doc1, ..., docN],    # full pool for recall; omit to skip recall
        },
        ...
    ]
    report = await evaluate_retrieval(llm, samples)
"""
import asyncio
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


@dataclass
class RetrievalReport:
    precision_mean: float
    recall_mean: float | None   # None if no sample provided all_docs
    samples: list[dict]


async def evaluate_retrieval(
    llm: BaseChatModel,
    samples: list[dict],
) -> RetrievalReport:
    """Evaluate retrieval quality for a list of query samples.

    Each sample dict:
        query    (str)              — the search query
        retrieved (list[Document]) — docs returned by the retriever
        all_docs  (list[Document]) — full pool of indexed docs (optional, enables recall)
    """
    results = []
    for sample in samples:
        query = sample["query"]
        retrieved: list[Document] = sample.get("retrieved", [])
        all_docs: list[Document] | None = sample.get("all_docs")

        if not retrieved:
            results.append({
                "query": query,
                "precision": 0.0,
                "recall": None,
                "retrieved_count": 0,
                "relevant_retrieved": 0,
            })
            continue

        relevance_flags = await asyncio.gather(
            *[_judge_relevance(llm, query, doc) for doc in retrieved]
        )
        relevant_retrieved = sum(relevance_flags)
        precision = relevant_retrieved / len(retrieved)

        recall: float | None = None
        if all_docs is not None:
            all_relevance = await asyncio.gather(
                *[_judge_relevance(llm, query, doc) for doc in all_docs]
            )
            total_relevant = sum(all_relevance)
            if total_relevant == 0:
                recall = 1.0  # nothing relevant to miss
            else:
                retrieved_contents = {d.page_content for d in retrieved}
                retrieved_relevant = sum(
                    rel
                    for doc, rel in zip(all_docs, all_relevance)
                    if doc.page_content in retrieved_contents
                )
                recall = retrieved_relevant / total_relevant

        results.append({
            "query": query,
            "precision": round(precision, 3),
            "recall": round(recall, 3) if recall is not None else None,
            "retrieved_count": len(retrieved),
            "relevant_retrieved": relevant_retrieved,
        })

    precision_vals = [r["precision"] for r in results]
    recall_vals = [r["recall"] for r in results if r["recall"] is not None]

    return RetrievalReport(
        precision_mean=round(sum(precision_vals) / len(precision_vals), 3) if precision_vals else 0.0,
        recall_mean=round(sum(recall_vals) / len(recall_vals), 3) if recall_vals else None,
        samples=results,
    )


async def _judge_relevance(llm: BaseChatModel, query: str, doc: Document) -> bool:
    prompt = f"""Does the following message contain information that is relevant to answering this query?

Query: {query}

Message: {doc.page_content}

Answer with ONLY "YES" or "NO"."""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip().upper().startswith("YES")
