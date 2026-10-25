# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import os, time, uuid, tempfile, logging
from dotenv import load_dotenv
import fitz
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# ---------------- OpenTelemetry (Jaeger via OTLP/HTTP) ----------------
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.trace import get_current_span

# ---------------- JSON logging (Loki-friendly) ----------------
from pythonjsonlogger import jsonlogger

# ===========================================================
#                  RAG + PLGJ OBSERVABILITY
# ===========================================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

# ---------- JSON Logger ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("rag-service")
logger.setLevel(logging.INFO)
_stream = logging.StreamHandler()
_file = logging.FileHandler("logs/app.log")
fmt = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(component)s %(otel_trace_id)s %(otel_span_id)s"
)
_stream.setFormatter(fmt)
_file.setFormatter(fmt)
logger.addHandler(_stream)
logger.addHandler(_file)

def _otel_ids():
    span = get_current_span()
    ctx = span.get_span_context() if span else None
    if not ctx or not ctx.is_valid:
        return {"otel_trace_id": None, "otel_span_id": None}
    # hex strings
    return {
        "otel_trace_id": format(ctx.trace_id, "032x"),
        "otel_span_id": format(ctx.span_id, "016x"),
    }

def log(level, msg, request: Request = None, component: str = None, **kw):
    extra = {"request_id": getattr(getattr(request, "state", None), "request_id", None),
             "component": component, **_otel_ids()}
    getattr(logger, level)(msg, extra=extra, **kw)

# ---------- FastAPI ----------
app = FastAPI(title="RAG Service with PLGJ Observability")

# CORS (optional)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response

# ---------- Prometheus ----------
Instrumentator().instrument(app).expose(app)  # /metrics

RAG_LATENCY = Histogram("rag_latency_seconds", "End-to-end /ask latency")
RAG_RETRIEVAL_LAT = Histogram("rag_retrieval_latency_seconds", "Vector search latency")
RAG_EMBED_LAT = Histogram("rag_query_embedding_latency_seconds", "Query embedding latency")
RAG_LLM_LAT = Histogram("rag_llm_latency_seconds", "LLM generation latency")
RAG_ERRORS = Counter("rag_errors_total", "RAG pipeline errors total")
RAG_DOCS = Histogram("rag_retrieved_docs", "Docs retrieved per query", buckets=(0,1,3,5,10))

# ---------- OpenTelemetry (Jaeger via OTLP/HTTP) ----------
resource = Resource(attributes={SERVICE_NAME: "rag-fastapi-service"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces"))
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

# ---------- RAG Core ----------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = os.getenv("PINECONE_INDEX", "startnew")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

model = ChatOpenAI(model="gpt-4o")
prompt = PromptTemplate(
    template=("You are a helpful assistant. Use the context to answer. If unsure, say you don't know.\n\n"
              "Context:\n{context}\n\nQuestion:\n{question}\n"),
    input_variables=["context", "question"],
)
parser = StrOutputParser()
chain = prompt | model | parser
chat_history = []

# ===========================================================
#                        ROUTES
# ===========================================================
@app.get("/health")
def health_check(request: Request):
    log("info", "Health check", request, component="health")
    return {"status": "healthy"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), request: Request = None):
    log("info", f"Upload received: {file.filename}", request, component="ingest")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            pdf = fitz.open(tmp_path)
            docs = []
            for i in range(len(pdf)):
                page = pdf.load_page(i)
                text = page.get_text()
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"page": i}))
            pdf.close()
        finally:
            os.remove(tmp_path)

        vector_store.add_documents(docs)
        log("info", f"Uploaded {len(docs)} documents", request, component="ingest")
        return {"status": "success", "uploaded_pages": len(docs)}
    except Exception as e:
        RAG_ERRORS.inc()
        log("error", f"Upload failed: {e}", request, component="ingest")
        return {"status": "error", "message": str(e)}

@app.post("/ask/")
def ask_question(question: str, request: Request):
    t0 = time.monotonic()
    with tracer.start_as_current_span("ask_question") as root_span:
        try:
            log("info", f"Question: {question}", request, component="api")
            chat_history.append({"role": "user", "content": question})

            # 1) Query embedding
            with tracer.start_as_current_span("query_embed") as span_embed:
                e0 = time.monotonic()
                query_vec = embeddings.embed_query(question)
                e1 = time.monotonic()
                RAG_EMBED_LAT.observe(e1 - e0)
                span_embed.set_attribute("embed.ms", int((e1 - e0) * 1000))

            # 2) Vector search
            with tracer.start_as_current_span("vector_search") as span_retr:
                r0 = time.monotonic()
                # Use by-vector to separate embed vs search latency
                docs = vector_store.similarity_search_by_vector(query_vec, k=3)
                r1 = time.monotonic()
                RAG_RETRIEVAL_LAT.observe(r1 - r0)
                RAG_DOCS.observe(len(docs))
                span_retr.set_attribute("rag.docs_count", len(docs))
                context = "\n".join([d.page_content for d in docs])

            # 3) LLM generation
            with tracer.start_as_current_span("llm_generation") as span_llm:
                g0 = time.monotonic()
                answer = chain.invoke({"context": context, "question": question})
                g1 = time.monotonic()
                RAG_LLM_LAT.observe(g1 - g0)
                span_llm.set_attribute("llm.duration_ms", int((g1 - g0) * 1000))

            chat_history.append({"role": "assistant", "content": answer})
            total = time.monotonic() - t0
            RAG_LATENCY.observe(total)
            root_span.set_attribute("rag.total_ms", int(total * 1000))

            log("info", "Answer generated", request, component="api")
            return {"answer": answer, "latency_seconds": round(total, 3)}
        except Exception as e:
            RAG_ERRORS.inc()
            root_span.record_exception(e)
            log("error", f"RAG error: {e}", request, component="api")
            return {"error": str(e)}
