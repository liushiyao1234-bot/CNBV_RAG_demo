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


QUESTION_PATH = "experiments/test_questions.json"
CHUNK_PATH = "data/processed/chunks_semantic.json"

OUTPUT_PATH = "outputs/reranker_diagnostic.json"

TEST_IDS = {
    "q1",
    "q4",
    "q12",
    "q15",
}

RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5


def load_questions():
    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    return [
        question
        for question in questions
        if question["id"] in TEST_IDS
    ]


def simplify_retrieval_results(
    results,
):
    simplified = []

    for result in results:

        chunk = result["chunk"]

        simplified.append({
            "retrieval_rank":
                result["rank"],

            "retrieval_score":
                result["score"],

            "chunk_id":
                chunk["chunk_id"],

            "pages":
                (
                    f"{chunk.get('start_page')}"
                    f" → "
                    f"{chunk.get('end_page')}"
                ),

            "text":
                chunk["text"],
        })

    return simplified


def simplify_rerank_results(
    results,
):
    simplified = []

    for result in results:

        chunk = result["chunk"]

        simplified.append({
            "rerank_rank":
                result["rerank_rank"],

            "original_retrieval_rank":
                result[
                    "retrieval_rank"
                ],

            "retrieval_score":
                result[
                    "retrieval_score"
                ],

            "reranker_score":
                result[
                    "reranker_score"
                ],

            "chunk_id":
                chunk["chunk_id"],

            "pages":
                (
                    f"{chunk.get('start_page')}"
                    f" → "
                    f"{chunk.get('end_page')}"
                ),

            "text":
                chunk["text"],
        })

    return simplified


def print_summary(
    question_id,
    query,
    retrieval_results,
    reranked_results,
):
    print(
        "\n"
        + "#" * 90
    )

    print(
        f"{question_id}: {query}"
    )

    print(
        "#" * 90
    )

    print(
        "\nRetriever Top 10:"
    )

    for result in retrieval_results:

        print(
            f"R{result['rank']:>2}  "
            f"chunk={result['chunk']['chunk_id']:<3}  "
            f"score={result['score']:.4f}"
        )

    print(
        "\nReranker Top 5:"
    )

    for result in reranked_results:

        print(
            f"R{result['rerank_rank']}  "
            f"chunk={result['chunk']['chunk_id']:<3}  "
            f"original=R"
            f"{result['retrieval_rank']:<2}  "
            f"rerank_score="
            f"{result['reranker_score']:.4f}"
        )


if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. 读取四个诊断问题
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"诊断问题数量: "
        f"{len(questions)}"
    )

    # --------------------------------------------------------
    # 2. Embedding 模型只加载一次
    # --------------------------------------------------------

    embedding_model = (
        load_embedding_model()
    )

    # --------------------------------------------------------
    # 3. Semantic chunks 只加载一次
    # --------------------------------------------------------

    chunks = load_chunks(
        CHUNK_PATH
    )

    # --------------------------------------------------------
    # 4. Passage embeddings 只算一次
    # --------------------------------------------------------

    chunk_embeddings = (
        build_chunk_embeddings(
            chunks=chunks,
            model=embedding_model,
        )
    )

    # --------------------------------------------------------
    # 5. Reranker 也只加载一次
    # --------------------------------------------------------

    reranker_model = (
        load_reranker()
    )

    experiment_results = []

    # --------------------------------------------------------
    # 6. 四个 query 依次跑
    # --------------------------------------------------------

    for question in questions:

        question_id = (
            question["id"]
        )

        query = (
            question["query"]
        )

        retrieval_results = retrieve(
            query=query,
            chunks=chunks,
            chunk_embeddings=(
                chunk_embeddings
            ),
            model=embedding_model,
            top_k=RETRIEVAL_TOP_K,
        )

        reranked_results = (
            rerank_retrieval_results(
                query=query,
                retrieval_results=(
                    retrieval_results
                ),
                model=reranker_model,
                top_k=RERANK_TOP_K,
            )
        )

        print_summary(
            question_id=question_id,
            query=query,
            retrieval_results=(
                retrieval_results
            ),
            reranked_results=(
                reranked_results
            ),
        )

        experiment_results.append({
            "id":
                question_id,

            "query":
                query,

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
    # 7. 保存完整结果
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
        "\n完整结果已保存到:"
    )

    print(
        OUTPUT_PATH
    )