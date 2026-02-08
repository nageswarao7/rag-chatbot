# RAG Chat Application

Production-ready, enterprise-grade RAG-based chat application with document ingestion, semantic chunking, and conversational AI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                                        │
│  │   Streamlit UI  │  ← Chat interface, document upload, source display     │
│  │   (Port 8501)   │                                                        │
│  └────────┬────────┘                                                        │
│           │ HTTP                                                            │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           ▼                        API LAYER                                │
│  ┌─────────────────┐                                                        │
│  │    FastAPI      │  ← REST API with CORS, logging, error handling        │
│  │   (Port 8000)   │                                                        │
│  │                 │                                                        │
│  │  Endpoints:     │                                                        │
│  │  POST /chat     │  → Conversational RAG queries                          │
│  │  POST /ingest   │  → Document upload & processing                        │
│  │  GET  /stats    │  → System statistics                                   │
│  │  GET  /health   │  → Health check for monitoring                         │
│  │  DELETE /clear  │  → Clear all data                                      │
│  └────────┬────────┘                                                        │
│           │                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────┤
│           ▼                      SERVICE LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RAG Service                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │Query Rewrite │→ │Query Decomp. │→ │   Parallel   │               │   │
│  │  │(Conversation)│  │(Sub-queries) │  │  Retrieval   │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │         │                                    │                       │   │
│  │         ▼                                    ▼                       │   │
│  │  ┌──────────────┐                    ┌──────────────┐               │   │
│  │  │   LRU Cache  │  ←──────────────── │   Reranker   │               │   │
│  │  │  (100 items) │                    │(Cross-Encoder)│               │   │
│  │  └──────────────┘                    └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Ingestion Service                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Parser     │→ │  Semantic    │→ │  Metadata    │               │   │
│  │  │(PDF/DOCX/TXT)│  │  Chunking    │  │  Enrichment  │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              DATA LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │    ChromaDB     │  │   Gemini LLM    │  │   Log Files     │             │
│  │  (Vector Store) │  │ (gemini-2.5-flash)│ │ (logs/*.log)   │             │
│  │  Persistent     │  │   + Embeddings  │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## RAG Pipeline Design

### 1. Query Processing
```
User Query: "Tell me more about it"
    │
    ▼ Query Rewriting (LLM)
"Tell me more about machine learning algorithms"
    │
    ▼ Query Decomposition (LLM)
["machine learning algorithms", "types of ML algorithms"]
    │
    ▼ Parallel Retrieval (ThreadPool)
[20+ candidate chunks from ChromaDB]
    │
    ▼ Cross-Encoder Reranking
[Top 5 most relevant chunks]
    │
    ▼ LLM Generation with Context
Final Answer + Sources
```

### 2. Semantic Chunking Strategy

**Why Semantic Chunking?**
| Approach | Description | Benefit |
|----------|-------------|---------|
| Embedding-based | Compute sentence embeddings | Understands meaning, not just text |
| Topic Detection | Compare adjacent sentence similarity | Splits at natural topic boundaries |
| Balanced Sizes | Target 1500 chars, max 3000 | Optimal for retrieval + context |

```
Document → Sentences → Embeddings → Similarity Matrix → Topic Breaks → Balanced Chunks
```

### 3. Caching Strategy
- **LRU Cache**: 100 items, 1-hour TTL
- **Cache Key**: Hash of (query + history_length)
- **Benefits**: Reduced LLM calls, faster repeated queries

### 4. Error Handling
- **Retry with Exponential Backoff**: 3 attempts, base delay 1s
- **Global Exception Handler**: Catches all unhandled errors
- **Graceful Degradation**: Returns meaningful error messages

## Quick Start

### 1. Prerequisites
```bash
# Python 3.10+
# Docker & Docker Compose (optional)
```

### 2. Set API Key
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. Run with Docker (Recommended)
```bash
docker-compose up --build
```
- **API**: http://localhost:8000
- **UI**: http://localhost:8501
- **Docs**: http://localhost:8000/docs

### 4. Run Locally
```bash
pip install -r requirements.txt

# Terminal 1: Backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/app.py
```

### 5. Ingest Documents
```bash
# upload via UI
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| GET | `/stats` | System statistics (chunks, cache) |
| POST | `/chat` | Send message with history |
| POST | `/ingest` | Upload document |
| DELETE | `/clear` | Delete all documents + cache |
| DELETE | `/cache` | Clear query cache only |

### Chat Request Example
```json
{
  "message": "What is machine learning?",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

## Azure Deployment

### Option 1: Azure Container Apps (Recommended)
```bash
# 1. Create resource group
az group create --name rag-chat-rg --location eastus

# 2. Create Container Registry
az acr create --name ragchatacr --resource-group rag-chat-rg --sku Basic
az acr login --name ragchatacr

# 3. Build and push images
docker build -t ragchatacr.azurecr.io/rag-backend:latest .
docker push ragchatacr.azurecr.io/rag-backend:latest

# 4. Create Container Apps environment
az containerapp env create \
  --name rag-chat-env \
  --resource-group rag-chat-rg \
  --location eastus

# 5. Deploy backend
az containerapp create \
  --name rag-backend \
  --resource-group rag-chat-rg \
  --environment rag-chat-env \
  --image ragchatacr.azurecr.io/rag-backend:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10 \
  --env-vars GEMINI_API_KEY=secretref:gemini-key
```

### Option 2: Azure Kubernetes Service (AKS)
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-backend
  template:
    spec:
      containers:
      - name: backend
        image: ragchatacr.azurecr.io/rag-backend:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Scaling & Production Considerations

### Horizontal Scaling
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Azure LB)                  │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Backend │   │ Backend │   │ Backend │   │ Backend │
    │ Pod 1   │   │ Pod 2   │   │ Pod 3   │   │ Pod N   │
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌──────────────┐          ┌──────────────┐
           │   ChromaDB   │          │   Redis      │
           │  (Persistent)│          │   (Cache)    │
           └──────────────┘          └──────────────┘
```

### Scaling Strategies
| Component | Strategy | Implementation |
|-----------|----------|----------------|
| API | Horizontal Pod Autoscaler | Scale 1-10 replicas based on CPU |
| Vector DB | Azure | Managed, scalable vector search |
| Cache | Azure Cache for Redis | Distributed caching |
| LLM | Rate limiting + queue | Prevent API overload |

### Monitoring
```yaml
# Prometheus metrics
- request_latency_seconds
- cache_hit_ratio
- llm_call_duration_seconds
- document_chunks_total
- error_count_total
```

### Cost Control
- **LLM Caching**: Reduces API calls by ~40%
- **Batch Ingestion**: Process documents in batches
- **Auto-scaling**: Scale down during low traffic
- **Reserved Instances**: 30-50% cost savings

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI app with endpoints
│   ├── config.py         # Configuration management
│   ├── rag_service.py    # RAG pipeline with caching
│   ├── ingestion.py      # Semantic chunking
│   ├── vector_store.py   # ChromaDB wrapper
│   └── cache.py          # LRU cache implementation
├── frontend/
│   └── app.py            # Streamlit chat UI
Batch ingestion
├── data/
│   ├── chromadb/         # Persistent vector store
│   └── documents/        # Document storage
├── logs/
│   └── rag_service.log   # Application logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env             # Create .env file and add your GOOGLE_API_KEY = "API_KEY" API key
```

## Technology Choices

| Component | Choice | Justification |
|-----------|--------|---------------|
| Backend | FastAPI | Async, fast, auto-docs, type hints |
| Frontend | Streamlit | Rapid prototyping, Python-native |
| Vector DB | ChromaDB | Simple, persistent, good for PoC |
| LLM | Gemini 2.5 Flash | Fast, large context, affordable |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, accurate |
| Reranker | ms-marco-MiniLM | Cross-encoder for precision |

## Security Considerations

- **API Key Management**: Environment variables, not in code
- **Input Validation**: Pydantic models, file type checking
- **CORS**: Configurable allowed origins
- **Rate Limiting**: Add via Azure API Management
- **Authentication**: Add OAuth2/JWT for production

## License

MIT License
