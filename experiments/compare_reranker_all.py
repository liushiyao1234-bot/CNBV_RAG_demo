import json
from pathlib import Path

from src.embedding import load_embedding_model
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
# 配置
# ============================================================

QUESTION_PATH = (
    "experiments/test_questions.json"
)

CHUNK_PATH = (
    "data/processed/"
    "chunks_semantic.json"
)

OUTPUT_PATH = (
    "outputs/"
    "reranker_all_questions.json"
)

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# 读取问题
# ============================================================

def load_questions():
    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(
            file
        )

    return questions


# ============================================================
# 页码显示
# ============================================================

def get_page_label(
    chunk,
):
    start_page = chunk.get(
        "start_page"
    )

    end_page = chunk.get(
        "end_page"
    )

    if (
        start_page is not None
        and end_page is not None
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
# 简化 Retriever 结果
# ============================================================

def simplify_retrieval_results(
    results,
):
    simplified = []

    for result in results:

        chunk = result[
            "chunk"
        ]

        simplified.append({
            "retrieval_rank":
                result["rank"],

            "retrieval_score":
                float(
                    result["score"]
                ),

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "pages":
                get_page_label(
                    chunk
                ),

            "char_count":
                chunk.get(
                    "char_count"
                ),

            "token_count":
                chunk.get(
                    "token_count"
                ),

            "break_reason":
                chunk.get(
                    "break_reason"
                ),

            "text":
                chunk[
                    "text"
                ],
        })

    return simplified


# ============================================================
# 简化 Reranker 结果
# ============================================================

def simplify_rerank_results(
    results,
):
    simplified = []

    for result in results:

        chunk = result[
            "chunk"
        ]

        simplified.append({
            "rerank_rank":
                result[
                    "rerank_rank"
                ],

            "original_retrieval_rank":
                result[
                    "retrieval_rank"
                ],

            "retrieval_score":
                float(
                    result[
                        "retrieval_score"
                    ]
                ),

            "reranker_score":
                float(
                    result[
                        "reranker_score"
                    ]
                ),

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "pages":
                get_page_label(
                    chunk
                ),

            "char_count":
                chunk.get(
                    "char_count"
                ),

            "token_count":
                chunk.get(
                    "token_count"
                ),

            "break_reason":
                chunk.get(
                    "break_reason"
                ),

            "text":
                chunk[
                    "text"
                ],
        })

    return simplified


# ============================================================
# PowerShell 紧凑打印
# ============================================================

def print_question_summary(
    question_id,
    query,
    retrieval_results,
    reranked_results,
):
    print(
        "\n"
        + "#" * 100
    )

    print(
        f"{question_id}: "
        f"{query}"
    )

    print(
        "#" * 100
    )

    print(
        "\nRetriever Top 10"
    )

    for result in (
        retrieval_results
    ):

        chunk = result[
            "chunk"
        ]

        print(
            f"R"
            f"{result['rank']:>2} | "
            f"chunk="
            f"{chunk['chunk_id']:<3} | "
            f"score="
            f"{result['score']:.4f} | "
            f"pages="
            f"{get_page_label(chunk)}"
        )

    print(
        "\nReranker Top 5"
    )

    for result in (
        reranked_results
    ):

        chunk = result[
            "chunk"
        ]

        print(
            f"R"
            f"{result['rerank_rank']} | "
            f"chunk="
            f"{chunk['chunk_id']:<3} | "
            f"from="
            f"R"
            f"{result['retrieval_rank']:<2} | "
            f"rerank="
            f"{result['reranker_score']:.4f} | "
            f"pages="
            f"{get_page_label(chunk)}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. 读取全部问题
    # --------------------------------------------------------

    questions = (
        load_questions()
    )

    print(
        f"\n问题数量: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # 2. Embedding 模型
    #    只加载一次
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # 3. Semantic chunks
    # --------------------------------------------------------

    chunks = load_chunks(
        CHUNK_PATH
    )

    print(
        f"\nSemantic Chunk 数量: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # 4. Passage embeddings
    #    只计算一次
    # --------------------------------------------------------

    print(
        "\n正在生成 "
        "Semantic Chunk Embeddings..."
    )

    chunk_embeddings = (
        build_chunk_embeddings(
            chunks=chunks,
            model=embedding_model,
        )
    )

    print(
        "Chunk Embeddings "
        "生成完成。"
    )

    # --------------------------------------------------------
    # 5. Reranker
    #    只加载一次
    # --------------------------------------------------------

    reranker_model = (
        load_reranker()
    )

    # --------------------------------------------------------
    # 6. 所有问题逐个运行
    # --------------------------------------------------------

    experiment_results = []

    for question in questions:

        question_id = (
            question[
                "id"
            ]
        )

        query = (
            question[
                "query"
            ]
        )

        # ----------------------------------------------------
        # Retriever
        # ----------------------------------------------------

        retrieval_results = (
            retrieve(
                query=query,
                chunks=chunks,
                chunk_embeddings=(
                    chunk_embeddings
                ),
                model=(
                    embedding_model
                ),
                top_k=(
                    RETRIEVAL_TOP_K
                ),
            )
        )

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        reranked_results = (
            rerank_retrieval_results(
                query=query,
                retrieval_results=(
                    retrieval_results
                ),
                model=(
                    reranker_model
                ),
                top_k=(
                    RERANK_TOP_K
                ),
            )
        )

        # ----------------------------------------------------
        # 打印紧凑结果
        # ----------------------------------------------------

        print_question_summary(
            question_id=(
                question_id
            ),
            query=query,
            retrieval_results=(
                retrieval_results
            ),
            reranked_results=(
                reranked_results
            ),
        )

        # ----------------------------------------------------
        # 保存完整结果
        # ----------------------------------------------------

        experiment_results.append({
            "id":
                question_id,

            "query":
                query,

            "retrieval_top_k":
                RETRIEVAL_TOP_K,

            "rerank_top_k":
                RERANK_TOP_K,

            "retrieval":
                simplify_retrieval_results(
                    retrieval_results
                ),

            "reranked":
                simplify_rerank_results(
                    reranked_results
                ),
        })

    # --------------------------------------------------------
    # 7. 输出 JSON
    # --------------------------------------------------------

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "全部实验完成。"
    )

    print(
        f"结果已保存到："
        f"{OUTPUT_PATH}"
    )

    print(
        "=" * 100
    )