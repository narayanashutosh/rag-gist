import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters.character import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Loading...")
    loader = TextLoader("./mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created{len(texts)} chunks.")

    embeddings = AzureOpenAIEmbeddings(
        # model=os.environ["AZURE_OPENAI_MODEL"],
        # azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
        # api_key=os.environ["AZURE_OPENAI_API_KEY"],
        # api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        # azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
    )

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Finished")
