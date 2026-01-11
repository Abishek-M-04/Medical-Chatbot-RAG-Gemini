from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load text file
with open("data/medical.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.create_documents([text])

# Embeddings (cloud)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Vector store
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print("✅ Medical data indexed successfully")