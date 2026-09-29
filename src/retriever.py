import json
from pathlib import Path

import numpy as np

from src.embedding import (
    load_embedding_model,
    embed_queries,
    embed_passages,
)


def load_chunks(chunk_path):
    """
    从 JSON 文件读取 chunks。
    """

    chunk_path = Path(chunk_path)

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"找不到 Chunk 文件: {chunk_path}"
        )

    with open(
        chunk_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return chunks


def build_chunk_embeddings(
    chunks,
    model
):
    """
    为所有 chunks 生成 passage embeddings。
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embed_passages(
        model=model,
        texts=texts
    )

    return embeddings


def retrieve(
    query,
    chunks,
    chunk_embeddings,
    model,
    top_k=5
):
    """
    根据 query 检索最相关的 Top-K chunks。
    """

    query_embedding = embed_queries(
        model=model,
        texts=[query]
    )[0]

    # 因为 query 和 passage embeddings
    # 都已经 normalize，
    # 点积就是 cosine similarity。
    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    # 从高到低排序
    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        chunk = chunks[index]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "chunk": chunk
        })

    return results


def print_results(
    query,
    results
):
    """
    把 retrieval 结果打印出来。
    """

    print(
        f"\nQuery:\n{query}"
    )

    print(
        "\n--- Top Retrieval Results ---"
    )

    for result in results:

        chunk = result["chunk"]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Rank: {result['rank']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        if "start_page" in chunk:

            print(
                f"Pages: "
                f"{chunk.get('start_page')} "
                f"→ "
                f"{chunk.get('end_page')}"
            )

        print(
            "\nText:"
        )

        print(
            chunk["text"]
        )


if __name__ == "__main__":

    chunk_path = (
        "data/processed/"
        "chunks_semantic.json"
    )

    query = (
        "¿Cuáles son las etapas "
        "del lavado de dinero?"
    )

    top_k = 5

    model = load_embedding_model()

    chunks = load_chunks(
        chunk_path
    )

    print(
        f"加载 Chunk 数量: "
        f"{len(chunks)}"
    )

    chunk_embeddings = (
        build_chunk_embeddings(
            chunks=chunks,
            model=model
        )
    )

    print(
        f"Chunk Embedding Shape: "
        f"{chunk_embeddings.shape}"
    )

    results = retrieve(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        model=model,
        top_k=top_k
    )

    print_results(
        query=query,
        results=results
    )