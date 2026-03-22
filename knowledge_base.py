from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
import glob
from dotenv import load_dotenv
load_dotenv()
from config import QDRANT_API_KEY


# client = chromadb.PersistentClient(path = "Database")

# collection = client.create_collection(
#     name="Algorithm", 
#     embedding_function=SentenceTransformerEmbeddingFunction()
# )

qdrant_client = QdrantClient(
    url="https://6d63122d-7b93-4b00-81e3-b454b3363930.eu-west-2-0.aws.cloud.qdrant.io", 
    api_key=QDRANT_API_KEY,
)

qdrant_client.recreate_collection(
    collection_name="Content",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # 384 is correct for MiniLM-L6-v2
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Add to Qdrant
qdrant = Qdrant(
    client=qdrant_client,
    collection_name="Content",  # Match the created collection (table) name 
    embeddings=embeddings,
)

pdf_files = glob.glob("PDFs/*.pdf")

for pdf_file in pdf_files:
    documents = PyPDFLoader(file_path=pdf_file).load()
    for i in range(len(documents)):
        qdrant.add_documents([documents[i]])


# documents = PyPDFLoader(file_path="Introduction to Algorithm .pdf").load()
# for i in range(len(documents)):
#     qdrant.add_documents([documents[i]])
