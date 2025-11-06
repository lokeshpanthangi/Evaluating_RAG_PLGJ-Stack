from fastapi import FastAPI, File, UploadFile, Request
import os
from dotenv import load_dotenv
import tempfile
import logging
import fitz
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from prometheus_fastapi_instrumentator import Instrumentator

# --- OpenTelemetry Imports ---
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ===========================================================
#                RAG + PGLJ INSTRUMENTED BACKEND
# ===========================================================

# 1️⃣ Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2️⃣ Logging Setup (for Loki)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)
logger = logging.getLogger("rag-service")

# 3️⃣ FastAPI app setup
app = FastAPI(title="RAG Service with PGLJ Observability")

# 4️⃣ Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# 5️⃣ OpenTelemetry (Tracing → Jaeger @ 13.126.98.205)
resource = Resource(attributes={
    SERVICE_NAME: "rag-fastapi-service"
})
trace_provider = TracerProvider(resource=resource)
trace_exporter = OTLPSpanExporter(
    endpoint="http://13.126.98.205:4318/v1/traces",  # Jaeger OTLP endpoint
)
trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# 6️⃣ RAG Core Components
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "startnew"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
model = ChatOpenAI(model="gpt-4o")
prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Use the following context to answer the user's question. "
        "If unsure, say you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}"
    ),
    input_variables=["context", "question"],
)
parser = StrOutputParser()
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
chain = prompt | model | parser

chat_history = []

# ===========================================================
#                      ROUTES
# ===========================================================

@app.get("/health")
def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        pdf_document = fitz.open(temp_file_path)
        documents = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            doc = Document(page_content=text, metadata={"page": page_num})
            documents.append(doc)
        pdf_document.close()

        vector_store.add_documents(documents)
        logger.info(f"Uploaded {len(documents)} documents to vector store.")
        return {"status": "success", "message": f"Uploaded {len(documents)} documents."}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/ask/")
def ask_question(question: str, request: Request):
    with tracer.start_as_current_span("ask_question") as span:
        try:
            logger.info(f"Question: {question}")
            chat_history.append({"role": "user", "content": question})

            docs = vector_store.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            span.set_attribute("rag.docs_count", len(docs))

            answer = chain.invoke({"context": context, "question": question})
            chat_history.append({"role": "assistant", "content": answer})

            logger.info("Answer generated successfully.")
            return {"answer": answer}

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            span.record_exception(e)
            return {"error": str(e)}
