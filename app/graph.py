import asyncio
import truststore
truststore.inject_into_ssl()


from mcp_client import read_mcp_file
from typing import TypedDict
from rag import get_retriever
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from evaluator import evaluate_rag



load_dotenv()
class AgentState(TypedDict):
    question: str
    context: list[str]
    answer: str
    evaluation: dict | None

# ============================================================
# RETRIVER AGENT
# ============================================================


async def retriever_agent(state: AgentState):
    question = state["question"]

    # ---------- RAG RETRIEVAL ----------
    retriever = get_retriever()

    documents = retriever.invoke(question)
    

    rag_context = [
        document.page_content
        for document in documents
    ]

    # ---------- MCP RETRIEVAL ----------
    mcp_file = None

    question_lower = question.lower()

    if "alpha" in question_lower:
        mcp_file = (
            r"C:\Users\GR131235\Desktop"
            r"\Enterprise-Knowledge-Assistant\mcp_data"
            r"\project_updates"
            r"\project_alpha_latest_update.txt"
        )

    elif (
        "maintenance" in question_lower
        or "announcement" in question_lower
    ):
        mcp_file = (
            r"C:\Users\GR131235\Desktop"
            r"\Enterprise-Knowledge-Assistant\mcp_data"
            r"\announcements"
            r"\company_announcement.txt"
        )

    elif (
        "system" in question_lower
        or "attendance" in question_lower
        or "portal" in question_lower
    ):
        mcp_file = (
            r"C:\Users\GR131235\Desktop"
            r"\Enterprise-Knowledge-Assistant\mcp_data"
            r"\operational_docs"
            r"\system_status.txt"
        )

    mcp_context = []

    if mcp_file:
        result = await read_mcp_file(mcp_file)

        for content in result.content:
            if hasattr(content, "text"):
                mcp_context.append(content.text)

    # ---------- COMBINE CONTEXT ----------
    combined_context = rag_context + mcp_context

    return {
        "context": combined_context
    }

def create_llm():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    return llm


# ============================================================
# RESPONSE AGENT
# ============================================================


def response_agent(state: AgentState):

    question = state["question"]
    context = state["context"]

  

    context_text = "\n\n".join(context)

    llm = create_llm()

    prompt = f"""
You are an enterprise knowledge assistant.

You MUST answer the user's question using the information
contained in the retrieved documents below.

IMPORTANT:
The retrieved documents are authoritative.

If the answer is present in the retrieved documents,
you MUST answer using that information.

Do NOT say that information is unavailable if the
information is present in the retrieved documents.

Do NOT use outside knowledge.

Do NOT invent facts.

User Question:
{question}

Retrieved Documents:
{context_text}

Now answer the question directly.

Answer:
""" 

    response = llm.invoke(prompt)

    answer = response.content.strip()

   

    print(answer)

    return {
        "answer": answer
    }


# ============================================================
# EVALUATOR AGENT
# ============================================================

async def evaluator_agent(state: AgentState):
    result = await evaluate_rag(
        question=state["question"],
        answer=state["answer"],
        contexts=state["context"],
    )

    return {
        "evaluation": result
    }


# ============================================================
# GRAPH BUILDING
# ============================================================


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_agent)
    graph.add_node("response", response_agent)
    graph.add_node("evaluator", evaluator_agent)
    # graph.add_node("retry_response", retry_response_agent)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "response")
    graph.add_edge("response", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()



# ============================================================
# MAIN FUNCTION
# ============================================================

async def main():

    app_graph = build_graph()

    while True:

        print("\n" + "=" * 70)
        print("       ENTERPRISE KNOWLEDGE ASSISTANT")
        print("=" * 70)

        question = input("\nEnter your question: ").strip()

        if not question:
            print("Please enter a valid question.")
            continue

        initial_state = {
            "question": question,
            "context": [],
            "answer": "",
            "evaluation": None,
        }

        result = await app_graph.ainvoke(initial_state)

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "=" * 70)
        print("RAGAS EVALUATION")
        print("=" * 70)
        print(result["evaluation"])

        while True:

            choice = input(
                "\nDo you want to ask another question? (y/n): "
            ).strip().lower()

            if choice in ("y", "yes"):
                break

            if choice in ("n", "no"):
                print(
                    "\nThank you for using Enterprise Knowledge Assistant."
                )
                return

            print("Please enter y or n.")


if __name__ == "__main__":
    asyncio.run(main())