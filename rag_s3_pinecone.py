import os
from dotenv import load_dotenv
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import boto3

def check_environment_variables():
    """Verify all required environment variables are set"""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'PINECONE_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def create_rag_chain():
    # Load and check environment variables
    load_dotenv()
    check_environment_variables()

    # Configure AWS credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'langchain-rag-demo')
    file_key = os.getenv('S3_FILE_KEY', 'python_basics.txt')

    print(f"Using AWS region: {aws_region}")
    print(f"Using S3 bucket: {bucket_name}")
    print(f"Using file key: {file_key}")

    try:
        # Initialize AWS session
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Load documents from S3
        loader = S3FileLoader(bucket=bucket_name, key=file_key, aws_session=session)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents from S3")
        
        # Print a sample of the content
        print("\nSample of loaded content:")
        print(documents[0].page_content[:500])

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"\nSplit documents into {len(chunks)} chunks")
        print("\nSample chunk:")
        print(chunks[0].page_content[:500])

        # Initialize embeddings
        embeddings = OpenAIEmbeddings()

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = "langchain-rag"

        # Check if index exists, if not create it
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"\nCreating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        
        # Get the index
        index = pc.Index(index_name)
        print(f"\nUsing existing Pinecone index: {index_name}")

        # Create vector store
        vectorstore = LangchainPinecone(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Add documents to the vector store
        print("\nAdding documents to vector store...")
        vectorstore.add_documents(chunks)
        print("Successfully added documents to vector store")

        # Create retriever with increased k value for more context
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Increased from 3 to 5 for more context
        )

        # Create custom prompt template with more explicit instructions
        template = """Use the following pieces of context to answer the question at the end. 
        The context contains theological content about God and religious concepts.
        If you don't know the answer based on the given context, just say that you don't know.
        Do not make up or infer information that isn't directly supported by the context.

        Context: {context}

        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create chain with temperature=0 for more precise answers
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return qa_chain

    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        raise

def ask_question(qa_chain, question: str):
    """
    Ask a question using the RAG chain
    
    Args:
        qa_chain: The retrieval QA chain
        question: The question to ask
        
    Returns:
        The answer from the chain
    """
    try:
        result = qa_chain({"query": question})
        print("\nQuestion:", question)
        print("\nAnswer:", result["result"])
        
        # Print source documents
        print("\nSource Documents:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
        return result
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return None

if __name__ == "__main__":
    try:
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
        
        # Test the chain with questions
        for question in test_questions:
            result = ask_question(qa_chain, question)
            print("\n" + "="*50 + "\n")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}") 