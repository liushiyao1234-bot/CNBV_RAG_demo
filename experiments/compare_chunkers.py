import json
from pathlib import Path

from src.embedding import load_embedding_model

from src.retriever import (
    load_chunks,
    build_chunk_embeddings,
    retrieve,
)


# ============================================================
# 实验配置
# ============================================================

TOP_K = 5

QUESTION_PATH = (
    "experiments/"
    "test_questions.json"
)

OUTPUT_PATH = (
    "outputs/"
    "retrieval_results.json"
)

CHUNK_SOURCES = {
    "fixed": (
        "data/processed/"
        "chunks_fixed.json"
    ),

    "recursive": (
        "data/processed/"
        "chunks_recursive.json"
    ),

    "semantic": (
        "data/processed/"
        "chunks_semantic.json"
    ),
}


# ============================================================
# 读取测试问题
# ============================================================

def load_questions(question_path):

    question_path = Path(
        question_path
    )

    if not question_path.exists():

        raise FileNotFoundError(
            f"找不到测试问题文件: "
            f"{question_path}"
        )

    with open(
        question_path,
        "r",
        encoding="utf-8",
    ) as file:

        questions = json.load(
            file
        )

    return questions


# ============================================================
# Page metadata
# ============================================================

def get_page_label(chunk):

    if (
        "start_page" in chunk
        and "end_page" in chunk
    ):

        start_page = chunk[
            "start_page"
        ]

        end_page = chunk[
            "end_page"
        ]

        if start_page == end_page:

            return str(
                start_page
            )

        return (
            f"{start_page} → "
            f"{end_page}"
        )

    if "page" in chunk:

        return str(
            chunk["page"]
        )

    return "Unknown"


# ============================================================
# 打印单次结果
# ============================================================

def print_method_results(
    method_name,
    results,
):

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"Method: "
        f"{method_name.upper()}"
    )

    print(
        "-" * 80
    )

    for result in results:

        chunk = result[
            "chunk"
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Rank: "
            f"{result['rank']}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Pages: "
            f"{get_page_label(chunk)}"
        )

        print(
            "\nText:"
        )

        print(
            chunk["text"]
        )


# ============================================================
# 保存结果
# ============================================================

def save_results(
    experiment_results,
    output_path,
):

    output_path = Path(
        output_path
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. 读取所有测试问题
    # --------------------------------------------------------

    questions = load_questions(
        QUESTION_PATH
    )

    print(
        f"测试问题数量: "
        f"{len(questions)}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    # --------------------------------------------------------
    # 2. 模型只加载一次
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # 3. 三套 chunks 及其 embeddings 也只生成一次
    # --------------------------------------------------------

    retrieval_indexes = {}

    for (
        method_name,
        chunk_path,
    ) in CHUNK_SOURCES.items():

        print(
            f"\n正在准备 "
            f"{method_name}..."
        )

        chunks = load_chunks(
            chunk_path
        )

        chunk_embeddings = (
            build_chunk_embeddings(
                chunks=chunks,
                model=model,
            )
        )

        retrieval_indexes[
            method_name
        ] = {
            "chunks":
                chunks,

            "embeddings":
                chunk_embeddings,
        }

        print(
            f"{method_name}: "
            f"{len(chunks)} chunks, "
            f"embedding shape "
            f"{chunk_embeddings.shape}"
        )

    # --------------------------------------------------------
    # 4. 对所有问题批量 retrieval
    # --------------------------------------------------------

    experiment_results = []

    for question in questions:

        question_id = question[
            "id"
        ]

        query = question[
            "query"
        ]

        print(
            "\n\n"
            + "#" * 90
        )

        print(
            f"{question_id}: "
            f"{query}"
        )

        print(
            "#" * 90
        )

        question_result = {
            "id":
                question_id,

            "query":
                query,

            "results":
                {},
        }

        for (
            method_name,
            index_data,
        ) in retrieval_indexes.items():

            results = retrieve(
                query=query,
                chunks=(
                    index_data[
                        "chunks"
                    ]
                ),
                chunk_embeddings=(
                    index_data[
                        "embeddings"
                    ]
                ),
                model=model,
                top_k=TOP_K,
            )

            print_method_results(
                method_name=(
                    method_name
                ),
                results=results,
            )

            method_results = []

            for result in results:

                chunk = result[
                    "chunk"
                ]

                method_results.append({
                    "rank":
                        result["rank"],

                    "score":
                        result["score"],

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
                        chunk["text"],
                })

            question_result[
                "results"
            ][
                method_name
            ] = method_results

        experiment_results.append(
            question_result
        )

    # --------------------------------------------------------
    # 5. 保存全部实验结果
    # --------------------------------------------------------

    save_results(
        experiment_results=(
            experiment_results
        ),
        output_path=OUTPUT_PATH,
    )

    print(
        "\n\n"
        + "=" * 90
    )

    print(
        "全部 Retrieval 实验完成。"
    )

    print(
        f"结果已保存到: "
        f"{OUTPUT_PATH}"
    )