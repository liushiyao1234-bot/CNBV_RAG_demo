# 有两个用处，一是把句子embedding后，可以判断相邻内容的语义是否发生变化，然后进行语义切片
# 第二是最终把整个chunk embedding后可以做查询召回


from sentence_transformers import SentenceTransformer


# 我们当前使用的本地 Embedding 模型
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_embedding_model():
    """
    加载 Embedding 模型。

    第一次运行时会从 Hugging Face 下载模型到本地缓存。
    后续运行通常直接读取本地缓存。
    """

    print(f"正在加载 Embedding 模型: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding 模型加载完成。")

    return model


def embed_for_similarity(model, texts):
    """
    将文本转换成 Embedding Vector。

    multilingual-e5 在做语义相似度任务时，
    推荐给输入文本加上 'query: ' 前缀。
    """

    prefixed_texts = [
        f"query: {text}"
        for text in texts
    ]

    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True
    )

    return embeddings


def embed_queries(
    model,
    texts
):
    """
    将用户问题编码成 query embedding。
    """

    prefixed_texts = [
        f"query: {text}"
        for text in texts
    ]

    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True
    )

    return embeddings


def embed_passages(
    model,
    texts
):
    """
    将文档 Chunk 编码成 passage embedding。
    """

    prefixed_texts = [
        f"passage: {text}"
        for text in texts
    ]

    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True
    )

    return embeddings






if __name__ == "__main__":

    # 先准备三句话。
    # 前两句语义比较接近，第三句明显是另一个话题。
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

    print("\nEmbedding 数量:")
    print(len(embeddings))

    print("\n每个 Embedding 的维度:")
    print(embeddings[0].shape)

    print("\n--- 第一句原文 ---")
    print(texts[0])

    print("\n--- 第一句 Embedding 的前 10 个数字 ---")
    print(embeddings[0][:10])

    print("\n--- 第二句 Embedding 的前 10 个数字 ---")
    print(embeddings[1][:10])

    print("\n--- 第三句 Embedding 的前 10 个数字 ---")
    print(embeddings[2][:10])