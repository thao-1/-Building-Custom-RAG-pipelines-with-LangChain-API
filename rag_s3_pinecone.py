import os
from dotenv import load_dotenv
from langchain_community.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import boto3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

def clean_html_content(text):
    """Clean HTML content and extract readable text"""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def check_environment_variables():
    """Verify all required environment variables are set"""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'PINECONE_API_KEY',
        'OPENAI_API_KEY',
        'S3_BUCKET_NAME',
        'S3_FILE_KEY',
        'AWS_REGION'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def create_rag_chain():
    """Create a RAG chain using S3FileLoader and Pinecone vector store"""
    
    # Load and check environment variables
    load_dotenv()
    check_environment_variables()

    # Configure AWS credentials
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    bucket_name = os.getenv('S3_BUCKET_NAME')
    file_key = os.getenv('S3_FILE_KEY')

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
        
        # Configure S3 client with specific region
        s3_config = {
            'region_name': aws_region
        }
        s3 = session.client('s3', **s3_config)
        
        # Verify bucket exists and is accessible
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"Successfully connected to bucket: {bucket_name}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                raise ValueError(f"Bucket {bucket_name} does not exist")
            elif error_code == '403':
                raise ValueError(f"Access denied to bucket {bucket_name}")
            else:
                raise ValueError(f"Error accessing bucket {bucket_name}: {str(e)}")

        # Verify file exists
        try:
            s3.head_object(Bucket=bucket_name, Key=file_key)
            print(f"Successfully verified file exists: {file_key}")
        except ClientError as e:
            raise ValueError(f"File {file_key} not found in bucket {bucket_name}")

        # Load documents from S3
        loader = S3FileLoader(
            bucket=bucket_name,
            key=file_key,
            aws_session=session
        )
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} documents from S3")
        
        # Clean HTML content
        for doc in documents:
            doc.page_content = clean_html_content(doc.page_content)
        
        # Print a sample of the cleaned content
        if documents:
            print("\nSample of loaded content (cleaned):")
            print(documents[0].page_content[:500])

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"\nSplit documents into {len(chunks)} chunks")
        
        if chunks:
            print("\nSample chunk:")
            print(chunks[0].page_content[:500])

        # Initialize embeddings
        embeddings = OpenAIEmbeddings()

        # Initialize Pinecone with new API
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "langchain-rag"

        # Create vector store using langchain's Pinecone wrapper
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=index_name,
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
            search_kwargs={"k": 5}
        )

        # Create custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
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
        print(f"Error in create_rag_chain: {str(e)}")
        return None

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
        if qa_chain is None:
            raise ValueError("QA chain is not initialized")
            
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
        
        if qa_chain is None:
            print("Failed to initialize QA chain. Please check the errors above.")
            exit(1)
        
        # Test questions 
        test_questions = [
            "What are the main topics covered in this textbook?",
            "What is the most important concept discussed in the text?",
            "Can you summarize the key themes of this document?",
            "How does the textbook describe God's relationship with humanity?",
        ]
        
        # Test the chain with questions
        for question in test_questions:
            result = ask_question(qa_chain, question)
            print("\n" + "="*50 + "\n")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")