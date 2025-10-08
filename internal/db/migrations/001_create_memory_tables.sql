-- Create required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Main table storing long-term memory entries
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    context_id TEXT,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    embedding_model TEXT,
    score DOUBLE PRECISION,
    importance DOUBLE PRECISION,
    ttl_seconds INTEGER,
    expires_at TIMESTAMPTZ,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_memory_entries_project_scope
    ON memory_entries (project_id, scope)
    WHERE is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_memory_entries_context
    ON memory_entries (context_id)
    WHERE context_id IS NOT NULL AND is_deleted = FALSE;

CREATE INDEX IF NOT EXISTS idx_memory_entries_updated_at
    ON memory_entries (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_entries_metadata_gin
    ON memory_entries USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_memory_entries_embedding
    ON memory_entries USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- Tags associated with memory entries
CREATE TABLE IF NOT EXISTS memory_tags (
    entry_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,
    PRIMARY KEY (entry_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag
    ON memory_tags (tag)
    WHERE is_deleted = FALSE;

-- Relationships between memory entries
CREATE TABLE IF NOT EXISTS memory_links (
    id BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_memory_links_project_type
    ON memory_links (project_id, link_type)
    WHERE is_deleted = FALSE;

-- History table to track edits and deletes
CREATE TABLE IF NOT EXISTS memory_entry_history (
    id BIGSERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES memory_entries(id) ON DELETE CASCADE,
    previous_version INTEGER,
    version INTEGER NOT NULL,
    operation TEXT NOT NULL,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536),
    score DOUBLE PRECISION,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    changed_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_entry_history_entry
    ON memory_entry_history (entry_id, version DESC);

