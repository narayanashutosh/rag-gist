import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore

load_dotenv()

print("Initializing components...")
embeddings = AzureOpenAIEmbeddings(model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"])

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    model=os.environ["AZURE_OPENAI_MODEL"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

vector_stores = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vector_stores.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:

    {context}

    Question: {question}
    Provide a detailed answer:
    """)


def format_docs(docs):
    """Format retrieve documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


# ==============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ==============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without addtional code
    - Harder to compose other chains
    - More verbose and error-prone
    """
    # Step 1: Retreive relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and query
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted message
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ==============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ==============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Languae).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.aiinvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is Pineconde in machin learning?"

    # ======================================================================
    # Option 0: Raw invocation without RAG
    # ======================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
    print("Retrieving...")

    # ======================================================================
    # Option 1: Use implmentation without LCEL
    # ======================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ======================================================================
    # Option 1: Use implmentation WITH LCEL (Better Approach)
    # ======================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
