# Custom RAG Pipelines with LangChain

This project implements three different Retrieval-Augmented Generation (RAG) pipelines using LangChain, each with a unique document loader and vector database.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with your API keys:
```
LANGCHAIN_API_KEY=your_langchain_api_key
OPENAI_API_KEY=your_openai_api_key  # Required for embeddings and LLM
PINECONE_API_KEY=your_pinecone_api_key
WEAVIATE_API_KEY=your_weaviate_api_key
```

## RAG Implementations

1. `rag_local_chroma.py`: Uses LocalFileLoader with Chroma DB
2. `rag_s3_pinecone.py`: Uses S3FileLoader with Pinecone
3. `rag_crawler_weaviate.py`: Uses RecursiveUrlLoader with Weaviate

## Usage

Each RAG implementation can be run independently:

```bash
python rag_local_chroma.py
python rag_s3_pinecone.py
python rag_crawler_weaviate.py
```

## Notes
- Each implementation demonstrates a different approach to document loading and vector storage
- Make sure to have appropriate API keys and access credentials set up
- The implementations use OpenAI's embeddings and LLM for consistency and quality