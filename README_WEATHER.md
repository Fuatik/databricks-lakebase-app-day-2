# Weather RAG Pipeline

End-to-end RAG pipeline: harvest weather data, generate embeddings, and perform semantic search with pgvector on Lakebase Postgres.

## Data Source: National Weather Service (NWS)

**Why NWS?**
* Free, no auth required
* Rich narrative text (alerts + forecasts) ideal for semantic search
* Real-time data from NOAA
* Geographic coverage across US

**Data fetched:**
* Active weather alerts (description, instructions)
* Multi-day detailed forecasts (narrative per period)

## Schema Decisions

### weather_documents
```sql
CREATE TABLE weather_documents (
    id              TEXT PRIMARY KEY,          -- <source_type>:<hash> for dedup
    location        TEXT NOT NULL,
    source_type     TEXT CHECK (source_type IN ('alert', 'forecast')),
    headline        TEXT,
    narrative_text  TEXT NOT NULL,             -- Primary text for embedding
    issued_at       TIMESTAMPTZ,
    payload         JSONB,                     -- Raw API response
    synced_at       TIMESTAMPTZ
);
```

### weather_embeddings
```sql
CREATE TABLE weather_embeddings (
    id              TEXT PRIMARY KEY,          -- <document_id>:<chunk_index>
    document_id     TEXT REFERENCES weather_documents(id),
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR(384) NOT NULL,      -- MiniLM-L6-v2 output
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_weather_embeddings_hnsw
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);    -- 40-100x speedup for 10k+ embeddings
```

**Key decisions:**
* **Deduplication:** Alert IDs from NWS, forecast SHA256 hashes
* **Chunking:** 800 chars with 100 overlap
* **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, normalized)

**Schema setup:** Tables are auto-created on first `/weather/sync` call via `lakebase.ensure_weather_documents_table()` and `lakebase.ensure_weather_embeddings_table()`.

## How to Run

### 1. Sync Weather Documents
```bash
curl -X POST http://localhost:8000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
```
Fetches alerts and forecasts, deduplicates, stores in `weather_documents`. Tables are created automatically on first run.

### 2. Generate Embeddings
Run `notebooks/ingest_weather_embeddings`:
* Chunks text (800 chars, 100 overlap)
* Embeds with MiniLM-L6-v2
* Batch inserts to `weather_embeddings` via psycopg2 (Lakebase doesn't support Spark JDBC writes)
* **Includes HNSW benchmark at the end**

### 3. Search
```bash
curl -X POST http://localhost:8000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "tornado warning", "top_k": 5}'
```
Optional: Add `"source_type": "alert"` or `"forecast"` to filter.





## Known Limitations & Improvements

**Current limitations:**
* Geocoding rate limits (Nominatim: 1 req/sec)
* No LLM answer synthesis (returns raw chunks only)
* Serial embedding processing (batch of 32)
* No TTL cleanup for old alerts/forecasts
* Pure vector search (no hybrid BM25 + vector)

**Future improvements:**
* Scheduled ingestion (Databricks Job, hourly)
* LLM-based answer synthesis from top-k chunks
* Hybrid search (semantic + keyword + metadata filters)
* Cross-encoder reranking for precision
* DiskANN or partitioning for >100k embeddings
* Query latency and cache observability