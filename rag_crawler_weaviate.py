import os
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import weaviate
from weaviate.connect import ConnectionParams

# Load environment variables
load_dotenv()

def create_rag_chain():
    # Initialize Weaviate client
    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.with_api_key(
            api_key=os.getenv("WEAVIATE_API_KEY"),
            host=os.getenv("WEAVIATE_URL")
        )
    )

    # Initialize document loader for web crawling
    loader = RecursiveUrlLoader(
        url="https://realpython.com/python-basics/",  # Real Python Python Basics Tutorial
        max_depth=2,
        extractor=lambda x: BeautifulSoup(x, "html.parser").get_text()
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from URL")

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

    # Create Weaviate schema if it doesn't exist
    class_name = "RAGDocument"
    schema = {
        "class": class_name,
        "vectorizer": "none",  # We'll use OpenAI embeddings
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "source",
                "dataType": ["text"],
            }
        ]
    }

    # Check if schema exists and create if it doesn't
    try:
        client.schema.get(class_name)
        print(f"Using existing Weaviate schema: {class_name}")
    except:
        client.schema.create_class(schema)
        print(f"Created new Weaviate schema: {class_name}")

    # Create vector store
    vectorstore = Weaviate(
        client=client,
        index_name=class_name,
        text_key="text",
        embedding=embeddings,
        by_text=False
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
    
    return qa_chain

def ask_question(qa_chain, question: str):
    """Ask a question and get an answer from the RAG system"""
    result = qa_chain({"query": question})
    print("\nQuestion:", question)
    print("\nAnswer:", result["result"])
    print("\nSources:", [doc.metadata for doc in result["source_documents"]])
    return result

if __name__ == "__main__":
    # Create the RAG chain
    qa_chain = create_rag_chain()
    
    # Test questions
    test_questions = [
        "What are the main topics covered in this document?",
        "What is the roadmap for Python learners?",
        "How does the web crawling process work?",
    ]
    
    # Test the RAG pipeline
    for question in test_questions:
        ask_question(qa_chain, question)
        print("\n" + "="*50 + "\n") 