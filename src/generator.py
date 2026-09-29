import os
import requests

from dotenv import load_dotenv


# ============================================================
# MaaS 配置
# ============================================================

MAAS_API_URL = (
    "https://api-ap-southeast-1."
    "modelarts-maas.com/v2/chat/completions"
)

MODEL_NAME = "glm-5.3"


# ============================================================
# 加载 API Key
# ============================================================

def load_maas_api_key():
    """
    从项目根目录 .env 中读取 MaaS API Key。
    """

    load_dotenv()

    api_key = os.getenv(
        "MAAS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "没有找到 MAAS_API_KEY。"
            "请检查项目根目录下的 .env 文件。"
        )

    return api_key


# ============================================================
# 构造 Context
# ============================================================

def build_context(
    contexts,
):
    """
    把经过筛选的多个 Chunk 拼成 LLM Context。

    contexts 的结构：

    [
        {
            "chunk_id": 8,
            "pages": "10",
            "text": "..."
        },
        ...
    ]

    除了正文，同时把 Chunk ID 和页码提供给 LLM，
    这样模型可以在最终答案中引用来源。
    """

    context_parts = []

    for context in contexts:

        chunk_id = context[
            "chunk_id"
        ]

        pages = context[
            "pages"
        ]

        text = context[
            "text"
        ]

        source_label = (
            f"Chunk {chunk_id}, "
            f"Page {pages}"
        )

        context_parts.append(
            f"[Source: {source_label}]\n"
            f"{text}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 调用 MaaS
# ============================================================

def generate_answer(
    query,
    contexts,
):
    """
    根据经过 Retriever + Reranker + Context Filter
    得到的 Context 调用 GLM-5.3。

    LLM 只能根据这些 Context 回答，
    并在答案中标注来源。
    """

    api_key = (
        load_maas_api_key()
    )

    context = build_context(
        contexts
    )

    system_prompt = (
        "You are a question-answering assistant "
        "for a retrieval-augmented generation system.\n\n"

        "Answer the user's question using only "
        "the provided sources.\n\n"

        "Rules:\n"

        "1. Do not use outside knowledge.\n"

        "2. If the provided sources do not contain "
        "enough information to answer the question, "
        "explicitly say that the provided context "
        "is insufficient.\n"

        "3. Answer directly and clearly.\n"

        "4. Preserve important terminology from "
        "the source material when appropriate.\n"

        "5. Answer in the same language as "
        "the user's question.\n"

        "6. Cite the source that supports the answer "
        "using exactly the source labels provided "
        "in the context.\n"

        "7. Use citation format like "
        "[Chunk 8, Page 10].\n"

        "8. Do not invent Chunk IDs, page numbers, "
        "or sources that were not provided.\n"

        "9. Place citations immediately after the "
        "sentence or paragraph they support."
    )

    user_prompt = (
        "Sources:\n\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}"
    )

    headers = {
        "Content-Type":
            "application/json",

        "Authorization":
            f"Bearer {api_key}",
    }

    data = {
        "model":
            MODEL_NAME,

        "stream":
            False,

        "messages": [
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ],
    }

    response = requests.post(
        MAAS_API_URL,
        headers=headers,
        json=data,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    answer = (
        result[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]
    )

    return answer


# ============================================================
# Generator 独立测试
# ============================================================

if __name__ == "__main__":

    query = (
        "¿Los reactivos piloto "
        "se consideran en el resultado "
        "de la prueba?"
    )

    test_contexts = [
        {
            "chunk_id": 20,
            "pages": "18",
            "text": (
                "Es importante mencionar que la prueba "
                "incluye un porcentaje de reactivos "
                "adicionales denominados reactivos piloto, "
                "los cuales son sometidos a un análisis "
                "estadístico que aporta información sobre "
                "su calidad técnica y de contenido, "
                "que no se consideran en el resultado."
            ),
        },
        {
            "chunk_id": 25,
            "pages": "20",
            "text": (
                "Todos los reactivos que conforman la "
                "prueba son de opción múltiple y están "
                "compuestos por cuatro opciones "
                "de respuesta."
            ),
        },
    ]

    answer = generate_answer(
        query=query,
        contexts=test_contexts,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "Question:"
    )

    print(
        query
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LLM Answer:"
    )

    print(
        answer
    )

    print(
        "=" * 80
    )