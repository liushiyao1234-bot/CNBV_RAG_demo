import numpy as np
import torch

from sentence_transformers import CrossEncoder


# ============================================================
# Reranker 配置
# ============================================================

RERANKER_MODEL_NAME = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)


# ============================================================
# 加载 Reranker
# ============================================================

def load_reranker():
    """
    加载轻量级 multilingual Cross-Encoder Reranker。

    相比 bge-reranker-v2-m3，
    这个模型更适合 CPU 环境。
    """

    print(
        f"正在加载 Reranker 模型: "
        f"{RERANKER_MODEL_NAME}"
    )

    model = CrossEncoder(
        RERANKER_MODEL_NAME,
        max_length=512,
        device="cpu",
    )

    print(
        "Reranker 模型加载完成。"
    )

    return model


# ============================================================
# Rerank Retriever Results
# ============================================================

def rerank_retrieval_results(
    query,
    retrieval_results,
    model,
    top_k=5,
):
    """
    对 Retriever 返回的候选 Chunk 重新排序。

    输入：
        query:
            用户问题

        retrieval_results:
            Retriever Top K

        model:
            CrossEncoder Reranker

        top_k:
            最终保留几个结果

    输出：
        带有 reranker_score 和 rerank_rank
        的重新排序结果。
    """

    # --------------------------------------------------------
    # 1. 构造 Query + Passage Pair
    # --------------------------------------------------------

    pairs = []

    for result in retrieval_results:

        chunk_text = (
            result[
                "chunk"
            ][
                "text"
            ]
        )

        pairs.append(
            [
                query,
                chunk_text,
            ]
        )

    # --------------------------------------------------------
    # 2. Cross-Encoder 推理
    #
    #    明确使用 Identity，
    #    取得模型原始 relevance logits。
    # --------------------------------------------------------

    raw_scores = model.predict(
        pairs,
        activation_fn=torch.nn.Identity(),
        show_progress_bar=False,
    )

    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    ).reshape(-1)

    # --------------------------------------------------------
    # 3. Sigmoid 转换
    #
    #    转成 0~1，
    #    这样 pipeline 里的 Context Filter
    #    仍然可以继续使用 score >= 0.2。
    #
    #    注意：
    #    这仍然不是“正确概率”。
    # --------------------------------------------------------

    scores = (
        1.0
        /
        (
            1.0
            + np.exp(
                -raw_scores
            )
        )
    )

    # --------------------------------------------------------
    # 4. 把 Score 放回原始结果
    # --------------------------------------------------------

    reranked_results = []

    for (
        result,
        reranker_score,
    ) in zip(
        retrieval_results,
        scores,
    ):

        reranked_results.append({
            "retrieval_rank":
                result[
                    "rank"
                ],

            "retrieval_score":
                result[
                    "score"
                ],

            "reranker_score":
                float(
                    reranker_score
                ),

            "chunk":
                result[
                    "chunk"
                ],
        })

    # --------------------------------------------------------
    # 5. 按 Reranker Score 排序
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda item:
            item[
                "reranker_score"
            ],

        reverse=True,
    )

    # --------------------------------------------------------
    # 6. 只保留最终 Top K
    # --------------------------------------------------------

    final_results = []

    for (
        rerank_rank,
        result,
    ) in enumerate(
        reranked_results[
            :top_k
        ],
        start=1,
    ):

        result[
            "rerank_rank"
        ] = rerank_rank

        final_results.append(
            result
        )

    return final_results