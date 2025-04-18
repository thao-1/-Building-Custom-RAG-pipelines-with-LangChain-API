import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def create_rag_chain():
    # Initialize document loader for the text file
    loader = TextLoader("textbook.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

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

    # Create and persist Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Created and persisted Chroma vector store")

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
        "What are the main teachings about God in this textbook?",
        "How does the textbook describe God's attributes and nature?",
        "Can you summarize the key theological concepts discussed in this book?",
        "What does the textbook say about God's relationship with humanity?",
        "What are the main biblical themes covered in this document?"
    ]
    
    # Test the RAG pipeline
    for question in test_questions:
        ask_question(qa_chain, question)
        print("\n" + "="*50 + "\n") 