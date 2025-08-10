# AI Coding Instructions for Peoples Algorithm Pipeline

## Architecture Overview

This is a distributed face recognition and clustering system designed for wedding photo processing. The system follows a microservices architecture with 4 main processing services orchestrated by Google Cloud Run Jobs:

```
Raw Images → Preprocessing → Vector Indexing → Clustering → Results
                ↓
            Real-time API (Pinecone + GCS)
```

## Core Services & Data Flow

### 1. Pipeline Executor (`algo_pipeline_executer/`)
- **Role**: Orchestrates the entire pipeline using Google Cloud Run Jobs
- **Key Pattern**: Sequential job execution with environment-based job name selection (`-prod` vs `-dev` suffix)
- **Entry Point**: `pipeline_executor.py` - triggers preprocessing → indexing → clustering → notification
- **Critical**: Uses `SESSION_KEY` for data isolation across customer sessions

### 2. Preprocessing Service (`preprocessing/`)
- **Purpose**: Face detection, embedding extraction, and quality scoring
- **Key Pattern**: Chunked parallel processing with intermediate GCS storage (see `preprocess_app.py`)
- **Outputs**: `embeddings.npy`, `faces.npy`, `original_paths.json`, `metrics.json` to `{session_key}/preprocess/`
- **Face Quality Scoring**: Multi-factor scoring system in `common/consts_and_utils.py` (sharpness, alignment, face count, etc.)

### 3. Vector Indexing Service (`vector_indexing/`)
- **Purpose**: Indexes face embeddings in Pinecone for real-time similarity search
- **Integration**: Uses Pinecone with euclidean distance (`PINECONE_*` constants in `common/consts_and_utils.py`)
- **Key Files**: `face_indexer.py` for Pinecone operations

### 4. Clustering Service (`clustering/`)
- **Purpose**: Groups faces by person using HDBSCAN clustering
- **Algorithm**: UMAP dimensionality reduction → HDBSCAN clustering → Face uniter for cluster merging
- **Key Pattern**: Wedding-optimized parameters (MIN_CLUSTER_SIZE_HDBSCAN=2 for guests with few photos)

### 5. Real-time API (`real_time/`)
- **Purpose**: FastAPI service for live selfie matching against processed albums
- **Endpoints**: `/process-image/` and `/retrieve-images/`
- **Pattern**: Direct Pinecone querying + GCS cluster retrieval

## Development Patterns

### Docker & Deployment
- **Local Development**: Use `Dockerfile.dev` for hot-reloading with volume mounts
- **Production**: Use `Dockerfile` for optimized builds
- **Deployment Script**: `./deploy.sh [prod|dev]` builds and pushes all services to Google Artifact Registry
- **Service Naming**: Environment suffix pattern (`clustering-job` vs `clustering-job-dev`)

### Data Storage Conventions
```
GCS Bucket Structure:
{session_key}/
├── raw/                    # Original uploaded images
├── preprocess/            # Embeddings, faces, metrics
├── preprocess/tmp/        # Intermediate chunked processing results
├── web/                   # Optimized web images
└── clusters/              # Final clustered results
```

### Configuration Management
- **Global Constants**: `common/consts_and_utils.py` contains all tuning parameters
- **Face Quality Weights**: Multiple scoring factors (sharpness: 0.35, alignment: 0.45, etc.)
- **Clustering Parameters**: Wedding-optimized HDBSCAN settings
- **Environment Variables**: `SESSION_KEY` for data isolation, `RUN_ENV` for prod/dev switching

### Error Handling & Logging
- **Pattern**: Structured logging with session context
- **Memory Management**: Explicit garbage collection in multiprocessing contexts
- **Chunked Processing**: Fail-safe intermediate storage for large datasets

## Key External Dependencies

- **Google Cloud**: Run Jobs, Storage, Artifact Registry
- **Pinecone**: Vector similarity search with euclidean distance
- **ML Stack**: OpenCV, NumPy, HDBSCAN, UMAP, FaceNet embeddings
- **Precompiled Wheels**: `precompiled_wheels/` and `pytorch_facenet_wheels/` for consistent dependencies

## Development Workflow

### Running Locally
```bash
# Build dev image for any service
docker build -f {service}/Dockerfile.dev -t peeps-{service}-local .

# Run with session isolation
docker run -e SESSION_KEY=test1 -v $(pwd)/{service}:/app/{service} peeps-{service}-local
```

### Testing Pipeline
1. Use `SESSION_KEY=test1` for consistent local testing
2. Monitor GCS bucket structure for proper data flow
3. Check service logs for memory usage and processing times

### Adding New Services
- Follow the `{service}/Dockerfile.dev` + `{service}/Dockerfile` pattern
- Add to `deploy.sh` with proper prod/dev job naming
- Integrate into `pipeline_executor.py` orchestration flow

## Common Pitfalls
- **Memory**: Face processing is memory-intensive - use chunked processing patterns
- **Session Isolation**: Always prefix GCS paths with `session_key`
- **Environment Switching**: Remember prod/dev job name suffixes in Cloud Run
- **Dependency Conflicts**: Use precompiled wheels for consistent builds across environments
