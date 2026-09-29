import numpy as np

from src.embedding import (
    load_embedding_model,
    embed_for_similarity
)


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度。

    返回值越接近 1，
    表示两个向量方向越接近，
    在 Embedding 场景下通常意味着语义越相似。
    """

    dot_product = np.dot(
        vector_a,
        vector_b
    )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    similarity = dot_product / (
        norm_a * norm_b
    )

    return float(similarity)


if __name__ == "__main__":

    texts = [
        "El lavado de dinero busca ocultar el origen ilícito de los fondos.",

        "El objetivo es hacer que recursos obtenidos ilegalmente parezcan legítimos.",

        "El examen tiene una duración máxima de cuatro horas."
    ]

    model = load_embedding_model()

    embeddings = embed_for_similarity(
        model=model,
        texts=texts
    )

    print("\n--- 测试句子 ---")

    for index, text in enumerate(texts, start=1):
        print(f"句子 {index}: {text}")

    similarity_1_2 = cosine_similarity(
        embeddings[0],
        embeddings[1]
    )

    similarity_1_3 = cosine_similarity(
        embeddings[0],
        embeddings[2]
    )

    similarity_2_3 = cosine_similarity(
        embeddings[1],
        embeddings[2]
    )

    print("\n--- 语义相似度 ---")

    print(
        f"\n句子 1 ↔ 句子 2: {similarity_1_2:.4f}"
    )
    print(f"1: {texts[0]}")
    print(f"2: {texts[1]}")

    print(
        f"\n句子 1 ↔ 句子 3: {similarity_1_3:.4f}"
    )
    print(f"1: {texts[0]}")
    print(f"3: {texts[2]}")

    print(
        f"\n句子 2 ↔ 句子 3: {similarity_2_3:.4f}"
    )
    print(f"2: {texts[1]}")
    print(f"3: {texts[2]}")