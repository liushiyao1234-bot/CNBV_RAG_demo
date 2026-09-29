from src.embedding import (
    load_embedding_model,
)

from src.retriever import (
    load_chunks,
    build_chunk_embeddings,
    retrieve,
)

from src.reranker import (
    load_reranker,
    rerank_retrieval_results,
)


# ============================================================
# 实验配置
# ============================================================

QUERY = (
    "¿Los reactivos piloto "
    "se consideran en el resultado "
    "de la prueba?"
)

CHUNK_PATH = (
    "data/processed/"
    "chunks_semantic.json"
)

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# 打印 Retriever 结果
# ============================================================

def print_retrieval_results(
    results,
):
    print(
        "\n\n"
        + "#" * 90
    )

    print(
        "RETRIEVER TOP 10"
    )

    print(
        "#" * 90
    )

    for result in results:

        chunk = result[
            "chunk"
        ]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Retrieval Rank: "
            f"{result['rank']}"
        )

        print(
            f"Retrieval Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

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


# ============================================================
# 打印 Reranker 结果
# ============================================================

def print_reranked_results(
    results,
):
    print(
        "\n\n"
        + "#" * 90
    )

    print(
        "RERANKER TOP 5"
    )

    print(
        "#" * 90
    )

    for result in results:

        chunk = result[
            "chunk"
        ]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Rerank Rank: "
            f"{result['rerank_rank']}"
        )

        print(
            f"Original Retrieval Rank: "
            f"{result['retrieval_rank']}"
        )

        print(
            f"Retrieval Score: "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        f"\nQuery:\n{QUERY}"
    )

    # --------------------------------------------------------
    # 1. 加载 Embedding 模型
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # 2. 加载 Semantic chunks
    # --------------------------------------------------------

    chunks = load_chunks(
        CHUNK_PATH
    )

    print(
        f"\n加载 Chunk 数量: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # 3. Passage Embeddings
    # --------------------------------------------------------

    chunk_embeddings = (
        build_chunk_embeddings(
            chunks=chunks,
            model=embedding_model,
        )
    )

    # --------------------------------------------------------
    # 4. Retriever 先召回 Top 10
    # --------------------------------------------------------

    retrieval_results = retrieve(
        query=QUERY,
        chunks=chunks,
        chunk_embeddings=(
            chunk_embeddings
        ),
        model=embedding_model,
        top_k=RETRIEVAL_TOP_K,
    )

    print_retrieval_results(
        retrieval_results
    )

    # --------------------------------------------------------
    # 5. 加载 Reranker
    # --------------------------------------------------------

    reranker_model = (
        load_reranker()
    )

    # --------------------------------------------------------
    # 6. 对 Retriever Top 10
    #    重新排序
    # --------------------------------------------------------

    reranked_results = (
        rerank_retrieval_results(
            query=QUERY,
            retrieval_results=(
                retrieval_results
            ),
            model=reranker_model,
            top_k=RERANK_TOP_K,
        )
    )

    print_reranked_results(
        reranked_results
    )