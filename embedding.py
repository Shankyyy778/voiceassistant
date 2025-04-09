import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import pinecone

# Set Pinecone API key directly in environment variable
os.environ["PINECONE_API_KEY"] = "pcsk_3qTZgK_EYEdZiqF7Biyyt1FVgEz6CtQbFEmS2jD7pjyRy6F29WWzA9pUpHVdbyT5qh3hxk"
os.environ["PINECONE_ENVIRONMENT"] = "gcp-starter"

def create_embeddings(pdf_path='INTERIA - COMPANY KNOWLEDGE BASE.pdf'):
    """Create embeddings from a PDF document and store in Pinecone."""
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(texts)}")
    
    # Initialize Pinecone and embeddings
    index_name = "hari2"
    api_key = os.environ["PINECONE_API_KEY"]
    
    # Initialize embeddings with correct import
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Initialize Pinecone client
    pinecone.init(api_key=api_key, environment=os.environ["PINECONE_ENVIRONMENT"])
    
    # Check if index exists and create if needed
    if index_name not in pinecone.list_indexes():
        print(f"Creating new Pinecone index: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=1024,  # mxbai-embed-large dimension
            metric="cosine"
        )
    else:
        print(f"Using existing Pinecone index: {index_name}")
    
    # Create embeddings and store in Pinecone in batches
    batch_size = 10  # Smaller batch size to prevent errors
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        try:
            LangchainPinecone.from_documents(
                batch, 
                embeddings, 
                index_name=index_name
            )
            print(f"Successfully processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
    
    print("Embeddings created and stored successfully!")

if __name__ == "__main__":
    create_embeddings() 