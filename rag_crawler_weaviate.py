import os
import ssl
from dotenv import load_dotenv
from langchain_community.document_loaders import BSHTMLLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.config import DataType
import time
import certifi

# Load environment variables
load_dotenv()

def create_rag_chain():
    """Create a RAG chain using RecursiveUrlLoader and Weaviate vector store"""
    
    # Check if WEAVIATE_API_KEY is set
    if not os.getenv("WEAVIATE_API_KEY"):
        raise ValueError("WEAVIATE_API_KEY must be set in .env file")
    
    # Print Weaviate version for debugging
    print(f"Using Weaviate client version: {weaviate.__version__}")
    
    # Fix for SSL certificate verification on macOS
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Try to connect to Weaviate with retries
    max_retries = 3
    retry_delay = 2  # seconds
    client = None
    
    for attempt in range(max_retries):
        try:
            print(f"Connecting to Weaviate (attempt {attempt+1}/{max_retries})...")
            # Connect to Weaviate Cloud Services
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url="https://uwbal7cfrxykokk30y5ukg.c0.us-west3.gcp.weaviate.cloud",
                auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
                headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Optional: if using OpenAI modules
                }
            )
            # Test connection
            client.is_ready()
            print("✅ Successfully connected to Weaviate!")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All connection attempts failed.")
                raise

    # Initialize document loader for local HTML file
    loader = BSHTMLLoader("realpython.html")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from local HTML file")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create Weaviate collection if it doesn't exist
    class_name = "PythonBasics"
    
    # Check if collection exists
    try:
        existing_collection = client.collections.get(class_name)
        print(f"✅ Using existing collection: {class_name}")
    except weaviate.exceptions.WeaviateCollectionNotFoundException:
        # Create collection with proper data types
        client.collections.create(
            name=class_name,
            vectorizer_config=None,  # Using external embeddings
            properties=[
                {"name": "text", "data_type": DataType.TEXT},
                {"name": "source", "data_type": DataType.TEXT}
            ]
        )
        print(f"✅ Created new collection: {class_name}")

    # Create vector store
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings
    )

    # Add documents to Weaviate
    vectorstore.add_documents(splits)
    print("Added documents to Weaviate vector store")

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create custom prompt template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0)

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain, client

def ask_question(qa_chain, question: str):
    """Ask a question and get an answer from the RAG system"""
    result = qa_chain({"query": question})
    print("\nQuestion:", question)
    print("\nAnswer:", result["result"])
    print("\nSources:", [doc.metadata for doc in result["source_documents"]])
    return result

if __name__ == "__main__":
    client = None
    try:
        # Create the RAG chain
        qa_chain, client = create_rag_chain()
        
        # Test questions
        test_questions = [
            "What are the main topics covered in Python basics?",
            "How do I install Python?",
            "What are Python variables and data types?",
            "How does control flow work in Python?",
        ]
        
        # Test the RAG pipeline
        for question in test_questions:
            ask_question(qa_chain, question)
            print("\n" + "="*50 + "\n")
    finally:
        # Close the Weaviate client connection
        if client is not None:
            client.close()