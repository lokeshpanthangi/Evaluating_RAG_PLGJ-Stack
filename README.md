# RAG System with PGLJ Stack

A Retrieval-Augmented Generation (RAG) application built with FastAPI and instrumented with the PGLJ observability stack (Prometheus, Grafana, Loki, Jaeger).

## Live Services

| Service | URL |
|---------|-----|
| **API Server** | http://15.206.211.26:8000/docs |
| **Grafana Monitoring** | http://13.126.98.205:3000/?orgId=1&from=now-6h&to=now&timezone=browser |

**Grafana Credentials:** admin / admin

---

## Overview

RAG system that allows PDF document upload and natural language querying with AI-generated answers. Fully instrumented with PGLJ stack for metrics, logs, and traces.

**Key Features:**
- PDF document processing with PyMuPDF
- Semantic search via Pinecone vector database
- OpenAI GPT-4 for answer generation
- Complete observability (Prometheus, Grafana, Loki, Jaeger)

---

## Architecture

### Two-Instance Deployment

#### **Instance 1: Application Server** (15.206.211.26)
- FastAPI application serving the RAG endpoints
- PyMuPDF (Fitz) for PDF text extraction
- Pinecone vector database for document embeddings
- OpenAI GPT-4 for answer generation
- Promtail agent for log shipping

#### **Instance 2: Monitoring Server** (13.126.98.205)
- Prometheus for metrics scraping
- Loki for log aggregation
- Grafana for unified visualization
- Jaeger for distributed tracing

```
┌─────────────────────────────────────┐
│   Instance 1: Application Server    │
│   (15.206.211.26:8000)              │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   FastAPI + RAG Service      │   │
│  │  - Upload PDFs               │   │
│  │  - Query Documents           │   │
│  │  - OpenAI GPT-4              │   │
│  │  - Pinecone Vector Store     │   │
│  └──────────────────────────────┘   │
│           │                         │
│           │ Logs/Metrics/Traces     │
│           ▼                         │
│  ┌──────────────────────────────┐   │
│  │      Promtail Agent          │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
            │
            │ Send to Monitoring Stack
            ▼
┌─────────────────────────────────────┐
│   Instance 2: Monitoring Server     │
│   (13.126.98.205)                   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   Prometheus (Port 9090)     │   │
│  │   - Scrapes metrics          │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   Loki (Port 3100)           │   │
│  │   - Aggregates logs          │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   Jaeger (Port 16686)        │   │ 
│  │   - Distributed tracing      │   │
│  └──────────────────────────────┘   │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   Grafana (Port 3000)        │   │
│  │   - Unified Dashboard        │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

---

## Tech Stack

### Application Layer
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | High-performance async API |
| **PDF Processing** | PyMuPDF (Fitz) | Text extraction from PDFs |
| **Vector Database** | Pinecone | Document embeddings storage |
| **Embeddings** | OpenAI text-embedding-3-small | Text vectorization (1024 dims) |
| **LLM** | OpenAI GPT-4 | Answer generation |
| **Orchestration** | LangChain | RAG pipeline management |

### Observability Stack (PGLJ)
| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| **Metrics** | Prometheus | 9090 | Time-series metrics collection |
| **Logs** | Loki | 3100 | Log aggregation |
| **Traces** | Jaeger | 16686 (UI), 4318 (OTLP) | Distributed tracing |
| **Visualization** | Grafana | 3000 | Unified monitoring dashboard |
| **Log Shipping** | Promtail | 9080 | Log forwarding to Loki |

### Instrumentation
- **prometheus-fastapi-instrumentator**: Automatic FastAPI metrics
- **OpenTelemetry**: Distributed tracing with OTLP export
- **Python Logging**: Structured application logs

---

## Project Structure

```
Deploy/
├── Instance_1_Application/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   ├── logs/
│   └── promptail/
│       └── promtail-config.yml
│
└── Instance_2_Monitoring/
    ├── docker-compose.yml
    ├── prometheus.yml
    └── loki-config.yml
```

---

## Installation

### Instance 1: Application Server

```bash
cd Instance_1_Application
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file:
```env
PINECONE_API_KEY=your_key
OPENAI_API_KEY=your_key
```

Run application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Instance 2: Monitoring Server

```bash
cd Instance_2_Monitoring
docker-compose up -d
```

---

## API Endpoints

### Base URL
```
http://15.206.211.26:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy"
}
```

#### 2. Upload PDF Document
```http
POST /uploadfile/
Content-Type: multipart/form-data
```
**Parameters:**
- `file`: PDF file (multipart/form-data)

**Response:**
```json
{
  "status": "success",
  "message": "Uploaded 25 documents."
}
```

#### 3. Ask Question
```http
POST /ask/
```
**Parameters:**
- `question`: string (query parameter)

**Response:**
```json
{
  "answer": "Based on the document content, the answer is..."
}
```

#### 4. Metrics (Prometheus)
```http
GET /metrics
```
Returns Prometheus-formatted metrics.

---

## Observability Stack

### Prometheus
- URL: http://13.126.98.205:9090
- Metrics: Request rates, latency, errors
- Scrape interval: 10 seconds

### Loki
- URL: http://13.126.98.205:3100
- Aggregates logs from application server
- Query: `{job="rag-app"}`

### Jaeger
- URL: http://13.126.98.205:16686
- Traces API requests and LLM calls
- OTLP endpoint: Port 4318

### Grafana
- URL: http://13.126.98.205:3000
- Unified dashboard for metrics, logs, and traces

---

## Usage Examples

### Example 1: Upload a PDF
```bash
curl -X POST "http://15.206.211.26:8000/uploadfile/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Example 2: Query the Document
```bash
curl -X POST "http://15.206.211.26:8000/ask/?question=What%20is%20the%20main%20topic%20of%20the%20document?" \
  -H "accept: application/json"
```

### Example 3: Check Metrics
```bash
curl http://15.206.211.26:8000/metrics
```

---

## Configuration

### Pinecone
- Index: `startnew`
- Embedding: text-embedding-3-small (1024 dims)
- Similarity: Cosine

### OpenAI
- Model: GPT-4
- Search: Top 3 chunks

---

## Troubleshooting

**Application not starting**
- Verify API keys in `.env`
- Check logs: `tail -f logs/app.log`

**Logs not in Grafana**
- Verify Promtail is running
- Test Loki: `curl http://13.126.98.205:3100/ready`

**Metrics missing**
- Check Prometheus targets: http://13.126.98.205:9090/targets
- Verify port 8000 is accessible

---

## License

MIT License

---

**Last Updated:** November 6, 2025

