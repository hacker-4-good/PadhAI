import chromadb 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction 
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
import glob


client = chromadb.PersistentClient(path = "DB")

collection = client.create_collection(
    name="Algorithm", 
    embedding_function=SentenceTransformerEmbeddingFunction()
)


pdf_files = glob.glob("PDFs/*.pdf")

for pdf_file in pdf_files:
    documents = PyPDFLoader(file_path=pdf_file).load()
    for i in range(len(documents)):
        collection.add(
            ids=f'{os.path.basename(pdf_file)}_id{i}', 
            documents=documents[i].page_content
        )
