from fastapi import FastAPI, File, UploadFile
import os
from dotenv import load_dotenv
import tempfile
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
import fitz
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()
app = FastAPI()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "startnew"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024)
model = ChatOpenAI(model="gpt-4o")
prompt = PromptTemplate(
    template="You are a helpful assistant that helps people find information. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. This is the Context i retrived from RAG : {context} ,  Question: {question}",
    input_variables=["context", "question"]
)
parser = StrOutputParser()
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

chain = prompt | model | parser

@app.get("/health")
def health_check():
    return {"status": "healthy"}



@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    #storing the file in a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Extract text from PDF using PyMuPDF (fitz)
    pdf_document = fitz.open(temp_file_path)
    documents = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        # Create document objects compatible with LangChain
        from langchain_core.documents import Document
        doc = Document(page_content=text, metadata={})
        documents.append(doc)
    pdf_document.close()
    
    # add the documents to the vector store
    vector_store.add_documents(documents)
    return {"status": "success", "message": f"Uploaded {len(documents)} documents."}


chat_history = []

@app.post("/ask/")
def ask_question(question: str):
    global chat_history
    chat_history.append({"role": "user", "content": question})
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = chain.invoke({"context": context, "question": question})
    chat_history.append({"role": "assistant", "content": answer})
    return {"answer": answer}