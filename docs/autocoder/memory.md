# Memory Backends & Operations Guide

Auto-Coder can persist agent memories in Redis (short-term) and PostgreSQL (long-term). This guide documents the
configuration knobs, readiness helpers, and troubleshooting steps operators need to keep both stores healthy.

## 🚀 Quick Start

1. Export the connection details for the stores you plan to use.
2. Run the setup helper to validate connectivity, apply migrations, and create search indexes:
   ```bash
   python scripts/setup_memory.py
   ```
3. Launch the manager or integration tests once the helper reports all steps as `READY`.

Use `--redis-only` or `--postgres-only` if you want to validate a single backend and `--migrations-dir` to point at
custom SQL bundles.

## 🔧 Environment Variables

Auto-Coder reads store configuration from `config.json` and the following environment variables:

| Variable | Purpose |
| --- | --- |
| `MEMORY_SHORT_TERM_BACKEND` / `MEMORY_LONG_TERM_BACKEND` | Select the backend for each scope (`memory`, `redis`, `postgres`). |
| `MEMORY_<SCOPE>_TTL_SECONDS` | Override default TTL for the chosen scope (`SHORT_TERM`, `LONG_TERM`, `COMBINED`). |
| `MEMORY_<SCOPE>_EMBEDDING_MODEL` | Force a specific embedding model for that scope. |
| `MEMORY_<SCOPE>_REDIS_URL` | Full Redis URL (`redis://user:pass@host:port/db`). |
| `MEMORY_<SCOPE>_REDIS_HOST` / `PORT` / `DB` | Hostname, port, and database number when URL is not provided. |
| `MEMORY_<SCOPE>_REDIS_USERNAME` / `PASSWORD` / `SSL` | Authentication options for Redis. |
| `MEMORY_<SCOPE>_POSTGRES_DSN` | Connection string for PostgreSQL. |
| `MEMORY_<SCOPE>_POSTGRES_HOST` / `PORT` / `DATABASE` / `USER` / `PASSWORD` | Individual PostgreSQL connection parts. |
| `MEMORY_<SCOPE>_POSTGRES_SSLMODE` | Optional SSL mode (e.g., `require`). |
| `MEMORY_REDIS_URL` / `MEMORY_POSTGRES_DSN` | Global fallbacks applied when per-scope overrides are missing. |

Testing helpers expect temporary services to be available and provide the following overrides:

| Variable | Purpose |
| --- | --- |
| `TEST_REDIS_URL` | Redis connection string used by `tests/test_memory_backends.py`. Required when `redis-server` is not installed locally. |
| `TEST_POSTGRES_DSN` | PostgreSQL DSN consumed by the test fixtures and the setup helper. Must point to a database with the `vector` and `uuid-ossp` extensions installed. |

## 🗄️ PostgreSQL Migrations & pgvector

SQL migrations live in `internal/db/migrations`. Running `python scripts/setup_memory.py` applies every `.sql` file in that
directory, ensuring the following objects exist:

- `memory_entries`, `memory_tags`, `memory_links`, and `memory_entry_history` tables.
- An IVFFLAT index on `memory_entries.embedding` for pgvector similarity search.
- Required extensions: `vector` (pgvector) and `uuid-ossp`.

If the helper reports missing extensions, install them as a superuser:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

Re-run the helper afterwards to confirm readiness.

## 📊 Redis Search Indexes

The Redis short-term store uses hashes with the pattern `memory:<scope>:record:<id>`. When a `vector_index` is configured in
`config.json` or via environment variables, the helper provisions a RediSearch index with:

- Text field for `content`.
- Tag fields for `tags`, `session_id`, and `agent_id`.
- Numeric field for `score`.
- Optional vector field (`embedding_blob`) using FLAT search with float32 payloads.

Ensure the Redis instance has the RediSearch module loaded. If `redis-server` lacks the module, the helper will report an
actionable error so you can install RediSearch or disable vector queries.

## 🧪 Testing the Backends

The new `tests/test_memory_backends.py` suite exercises lifecycle flows, promotion/compaction logic, and RAG retrieval across
Redis and PostgreSQL. To run these integration tests locally:

```bash
export TEST_REDIS_URL=redis://localhost:6379/0        # or rely on redis-server being on PATH
export TEST_POSTGRES_DSN=postgresql://user:pass@localhost/autocoder
pytest tests/test_memory_backends.py
```

The fixtures will start an ephemeral Redis instance when the binary is available. PostgreSQL tests require an existing
database with the pgvector extension installed.

## 🛠️ Troubleshooting

| Symptom | Suggested Fix |
| --- | --- |
| `redis-py is not installed` | Install the dependency (`pip install redis>=5.0`) or pin it in your environment. |
| `RedisSearch module is unavailable` | Load the RediSearch module on the target Redis server (`--loadmodule` or Docker image with RediSearch). |
| `database missing extensions: vector` | Install pgvector in your PostgreSQL instance and re-run the helper. |
| `no SQL migrations found` | Verify the repository checkout includes `internal/db/migrations` or pass `--migrations-dir`. |
| Integration tests skipped | Provide `TEST_REDIS_URL` / `TEST_POSTGRES_DSN` so fixtures can connect to real services. |

Keep the helper and tests in your CI pipeline to guarantee that production environments and development workstations remain in
sync.
