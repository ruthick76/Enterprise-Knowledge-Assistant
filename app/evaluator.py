import os
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langchain_ollama import OllamaEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings import BaseRagasEmbedding
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
)


load_dotenv()


# ============================================================
# OLLAMA EMBEDDING ADAPTER
# ============================================================

class OllamaRagasEmbedding(
    BaseRagasEmbedding
):

    def __init__(
        self,
        model="nomic-embed-text:latest",
    ):

        super().__init__()

        self.ollama = OllamaEmbeddings(
            model=model
        )

    def embed_text(
        self,
        text: str,
        **kwargs,
    ):

        return self.ollama.embed_query(
            text
        )

    async def aembed_text(
        self,
        text: str,
        **kwargs,
    ):

        return await self.ollama.aembed_query(
            text
        )


# ============================================================
# RAGAS LLM
# ============================================================

def create_ragas_llm():

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:

        raise ValueError(
            "GROQ_API_KEY is not set in .env"
        )

    groq_client = AsyncOpenAI(

        api_key=groq_api_key,

        base_url=(
            "https://api.groq.com/openai/v1"
        ),
    )

    return llm_factory(

        "openai/gpt-oss-20b",

        provider="openai",

        client=groq_client,

        temperature=0,
    )


# ============================================================
# RAGAS EMBEDDINGS
# ============================================================

def create_ragas_embeddings():

    return OllamaRagasEmbedding(

        model="nomic-embed-text:latest"
    )


# ============================================================
# SAFE SCORE CONVERSION
# ============================================================

def extract_score(result):

    try:

        value = result.value

        if value is None:

            return 0.0

        value = float(value)

        # Prevent invalid values
        if value < 0:

            return 0.0

        if value > 1:

            return 1.0

        return value

    except Exception as e:

        print(
            f"Could not extract RAGAS score: {e}"
        )

        return 0.0


# ============================================================
# EVALUATE RAG
# ============================================================

async def evaluate_rag(
    question: str,
    answer: str,
    contexts: list[str],
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RAGAS EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Answer: {answer}"
    )

    print(
        f"Context count: {len(contexts)}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not contexts:

        print(
            "No retrieved contexts."
        )

        return {

            "faithfulness": 0.0,

            "answer_relevancy": 0.0,
        }

    if not answer.strip():

        print(
            "Empty answer."
        )

        return {

            "faithfulness": 0.0,

            "answer_relevancy": 0.0,
        }

    # --------------------------------------------------------
    # Create evaluator
    # --------------------------------------------------------

    evaluator_llm = create_ragas_llm()

    embeddings = create_ragas_embeddings()

    # --------------------------------------------------------
    # Faithfulness
    # --------------------------------------------------------

    try:

        faithfulness_metric = Faithfulness(
            llm=evaluator_llm
        )

        faithfulness_result = (
            await faithfulness_metric.ascore(

                user_input=question,

                response=answer,

                retrieved_contexts=contexts,
            )
        )

        faithfulness = extract_score(
            faithfulness_result
        )

    except Exception as e:

        print(
            "\nFaithfulness evaluation failed:"
        )

        print(e)

        faithfulness = 0.0

    # --------------------------------------------------------
    # Answer Relevancy
    # --------------------------------------------------------

    try:

        answer_relevancy_metric = (
            AnswerRelevancy(

                llm=evaluator_llm,

                embeddings=embeddings,
            )
        )

        answer_relevancy_result = (
            await answer_relevancy_metric.ascore(

                user_input=question,

                response=answer,
            )
        )

        answer_relevancy = extract_score(
            answer_relevancy_result
        )

    except Exception as e:

        print(
            "\nAnswer relevancy evaluation failed:"
        )

        print(e)

        answer_relevancy = 0.0

    # --------------------------------------------------------
    # Print scores
    # --------------------------------------------------------

    print(
        "\nFaithfulness:",
        faithfulness
    )

    print(
        "Answer Relevancy:",
        answer_relevancy
    )

    print(
        "=" * 70
    )

    return {

        "faithfulness": faithfulness,

        "answer_relevancy": answer_relevancy,
    }