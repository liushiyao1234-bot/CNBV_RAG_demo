import time


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

from src.generator import (
    generate_answer,
)


# ============================================================
# 配置
# ============================================================

CHUNK_PATH = (
    "data/processed/"
    "chunks_semantic.json"
)

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5

CONTEXT_SCORE_THRESHOLD = 0.2
MAX_CONTEXT_CHUNKS = 3


# ============================================================
# 页码显示
# ============================================================

def get_page_label(
    chunk,
):
    """
    从 Chunk Metadata 中取得页码。
    """

    start_page = chunk.get(
        "start_page"
    )

    end_page = chunk.get(
        "end_page"
    )

    if (
        start_page is not None
        and
        end_page is not None
    ):

        if (
            start_page
            == end_page
        ):
            return str(
                start_page
            )

        return (
            f"{start_page}"
            f" → "
            f"{end_page}"
        )

    page = chunk.get(
        "page"
    )

    if page is not None:
        return str(
            page
        )

    return "?"


# ============================================================
# 初始化整个 RAG
# ============================================================

def initialize_rag():
    """
    一次性初始化：

    1. Embedding 模型
    2. Semantic Chunks
    3. Passage Embeddings
    4. Reranker
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "正在初始化 RAG..."
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 1. Embedding 模型
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # 2. Semantic Chunks
    # --------------------------------------------------------

    chunks = load_chunks(
        CHUNK_PATH
    )

    print(
        f"\n已加载 Semantic Chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # 3. Passage Embeddings
    # --------------------------------------------------------

    print(
        "\n正在生成 Passage Embeddings..."
    )

    chunk_embeddings = (
        build_chunk_embeddings(
            chunks=chunks,
            model=embedding_model,
        )
    )

    print(
        "Passage Embeddings 生成完成。"
    )

    # --------------------------------------------------------
    # 4. Reranker
    # --------------------------------------------------------

    reranker_model = (
        load_reranker()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RAG 初始化完成。"
    )

    print(
        "=" * 80
    )

    return {
        "embedding_model":
            embedding_model,

        "chunks":
            chunks,

        "chunk_embeddings":
            chunk_embeddings,

        "reranker_model":
            reranker_model,
    }


# ============================================================
# Context Filter
# ============================================================

def select_context(
    reranked_results,
):
    """
    从 Reranker Top 5 中挑选真正发送给 LLM 的 Context。

    当前规则：

    1. Rank 1 永远保留
    2. 其他结果 reranker_score >= 0.2 才保留
    3. 最多保留 3 个 Chunk
    """

    selected_results = []

    for result in reranked_results:

        if (
            len(selected_results)
            >= MAX_CONTEXT_CHUNKS
        ):
            break

        rerank_rank = (
            result[
                "rerank_rank"
            ]
        )

        reranker_score = (
            result[
                "reranker_score"
            ]
        )

        if (
            rerank_rank == 1
            or
            reranker_score
            >= CONTEXT_SCORE_THRESHOLD
        ):

            selected_results.append(
                result
            )

    return selected_results


# ============================================================
# 转换成 Generator 所需 Context
# ============================================================

def build_generator_contexts(
    selected_results,
):
    """
    把 Reranker Result 转换成 generator.py
    所需要的结构。

    同时带上 Chunk ID 和页码，
    供 LLM 生成引用。
    """

    contexts = []

    for result in selected_results:

        chunk = result[
            "chunk"
        ]

        contexts.append({
            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "pages":
                get_page_label(
                    chunk
                ),

            "text":
                chunk[
                    "text"
                ],
        })

    return contexts


# ============================================================
# 单次完整 RAG 查询
# ============================================================

def answer_with_rag(
    query,
    rag,
):
    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Retriever
    # --------------------------------------------------------

    start = time.perf_counter()

    retrieval_results = retrieve(
        query=query,
        chunks=rag["chunks"],
        chunk_embeddings=rag["chunk_embeddings"],
        model=rag["embedding_model"],
        top_k=RETRIEVAL_TOP_K,
    )

    retrieval_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # 2. Reranker
    # --------------------------------------------------------

    start = time.perf_counter()

    reranked_results = (
        rerank_retrieval_results(
            query=query,
            retrieval_results=(
                retrieval_results
            ),
            model=(
                rag["reranker_model"]
            ),
            top_k=RERANK_TOP_K,
        )
    )

    reranker_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # 3. Context Filter
    # --------------------------------------------------------

    start = time.perf_counter()

    selected_context_results = (
        select_context(
            reranked_results
        )
    )

    generator_contexts = (
        build_generator_contexts(
            selected_context_results
        )
    )

    context_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # 4. LLM
    # --------------------------------------------------------

    start = time.perf_counter()

    answer = generate_answer(
        query=query,
        contexts=generator_contexts,
    )

    generation_time = (
        time.perf_counter()
        - start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    return {
        "answer":
            answer,

        "retrieval_results":
            retrieval_results,

        "reranked_results":
            reranked_results,

        "selected_context_results":
            selected_context_results,

        "timing": {
            "retrieval":
                retrieval_time,

            "reranker":
                reranker_time,

            "context_filter":
                context_time,

            "generation":
                generation_time,

            "total":
                total_time,
        },
    }


# ============================================================
# 只打印来源 Metadata
# ============================================================

def print_context_sources(
    results,
):
    """
    只显示最终发送给 LLM 的来源信息。

    不再打印原始 Chunk 内容。
    """

    print(
        "\n"
        + "-" * 80
    )

    print(
        "Sources Sent to LLM:"
    )

    print(
        "-" * 80
    )

    for result in results:

        chunk = result[
            "chunk"
        ]

        page_label = (
            get_page_label(
                chunk
            )
        )

        print(
            f"[Chunk "
            f"{chunk['chunk_id']}, "
            f"Page "
            f"{page_label}] "
            f"rerank_score="
            f"{result['reranker_score']:.4f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. 初始化一次
    # --------------------------------------------------------

    rag = (
        initialize_rag()
    )

    print(
        "\n现在可以开始提问。"
    )

    print(
        "中文 / English / Español 都可以。"
    )

    print(
        "输入 exit 退出。"
    )

    # --------------------------------------------------------
    # 2. 循环提问
    # --------------------------------------------------------

    while True:

        query = input(
            "\nQuestion: "
        ).strip()

        if not query:
            continue

        if (
            query.lower()
            in {
                "exit",
                "quit",
                "q",
            }
        ):

            print(
                "\nRAG 已退出。"
            )

            break

        print(
            "\n正在检索..."
        )

        # ----------------------------------------------------
        # 3. 完整 RAG
        # ----------------------------------------------------

        result = (
            answer_with_rag(
                query=query,
                rag=rag,
            )
        )

        # ----------------------------------------------------
        # 4. 最终答案
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 80
        )

        print(
            "Answer:"
        )

        print(
            "=" * 80
        )

        print(
            result[
                "answer"
            ]
        )

        # ----------------------------------------------------
        # 5. 只显示真正发送给 LLM 的来源
        # ----------------------------------------------------

        print_context_sources(
            result[
                "selected_context_results"
            ]
        )

        timing = result[
            "timing"
        ]

        print(
            "\n"
            + "-" * 80
        )

        print(
            "Timing:"
        )

        print(
            "-" * 80
        )

        print(
            f"Retriever:      "
            f"{timing['retrieval']:.2f}s"
        )

        print(
            f"Reranker:       "
            f"{timing['reranker']:.2f}s"
        )

        print(
            f"Context Filter: "
            f"{timing['context_filter']:.4f}s"
        )

        print(
            f"LLM Generation: "
            f"{timing['generation']:.2f}s"
        )

        print(
            f"Total:          "
            f"{timing['total']:.2f}s"
        )