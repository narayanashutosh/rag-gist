import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone.vectorstores import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializing components...")
embeddings = AzureOpenAIEmbeddings(model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"])

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    model=os.environ["AZURE_OPENAI_MODEL"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)

vector_stores = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vector_stores.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:

    {context}

    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieve documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

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

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is Pineconde in machin learning?"

    # ====================================================================
    # Option 0: Raw invocation without RAG
    # ====================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
    print("Retrieving...")

    # ====================================================================
    # Option 1: Use implmentation without LCEL
    # ====================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)
