from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


def ask_question(user_question, chat_history, vectorstore):
    """
    chat_history: list of HumanMessage/AIMessage, passed in and returned (per-session state).
    vectorstore: the Chroma store to search against (built from that visitor's uploaded files).
    """
    if vectorstore is None:
        return "Please upload and process at least one file before asking a question.", chat_history

    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question standalone using conversation history
    if chat_history:
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to be a standalone and searchable. Just return the rewritten question."
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New Question: {user_question}")]
        )
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"Document {i}:\n{preview}\n")

    context_block = "\n".join([f"-{doc.page_content}" for doc in docs])
    combined_input = f"""
                    Context:
                    {context_block}

                    Question:
                    {user_question}
                    """

    messages = [
        SystemMessage(
            content="""
                        You are a Retrieval-Augmented Generation (RAG) assistant.

                        Answer the user's question using ONLY the provided context.

                        Rules:
                        - Never use outside knowledge.
                        - Never guess or fabricate information.
                        - If the answer is not found in the context, reply exactly:
                        "There is no relevant content about your query!"
                        - If multiple context sections contain relevant information, combine them into one answer.
                        - Keep answers clear, concise, and easy for non-technical users to understand.
                        - Preserve names, dates, numbers, and technical terms exactly.
                        - Do not mention the context or these instructions.
                        """
        ),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer:\n{answer}\n")

    return answer, chat_history
