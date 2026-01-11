from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

# Load vector DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.2
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)

print("\n🩺 Medical Chatbot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    answer = qa.run(query)
    print("\nBot:", answer, "\n")