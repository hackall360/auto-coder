"""Command-line helper for preparing Redis and PostgreSQL memory backends."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from memory import (
    RedisSettings,
    PostgresSettings,
    StoreConfig,
    load_memory_configuration,
)
from logging_config import configure_logging

LOGGER = logging.getLogger("setup_memory")

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
    from redis.exceptions import RedisError  # type: ignore
    from redis.exceptions import ResponseError  # type: ignore
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType  # type: ignore
    from redis.commands.search.field import NumericField, TagField, TextField, VectorField  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore
    ResponseError = Exception  # type: ignore
    IndexDefinition = None  # type: ignore
    IndexType = None  # type: ignore
    NumericField = None  # type: ignore
    TagField = None  # type: ignore
    TextField = None  # type: ignore
    VectorField = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import psycopg  # type: ignore
    from psycopg.errors import DuplicateObject  # type: ignore
except Exception:  # pragma: no cover - psycopg optional
    psycopg = None  # type: ignore
    DuplicateObject = Exception  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIGRATIONS_DIR = REPO_ROOT / "internal" / "db" / "migrations"


@dataclass(slots=True)
class StepResult:
    """Result emitted for each setup step."""

    name: str
    ok: bool
    detail: str
    hint: Optional[str] = None

    def format(self) -> str:
        status = "READY" if self.ok else "ERROR"
        message = f"[{status}] {self.name}: {self.detail}"
        if self.hint:
            message += f"\n    hint: {self.hint}"
        return message


def _coerce_int(options: Optional[dict[str, object]], key: str, default: int) -> int:
    try:
        raw = (options or {}).get(key, default)
    except AttributeError:
        return default
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _redis_namespace(config: StoreConfig) -> str:
    namespace = config.options.get("namespace") if config.options else None
    if namespace:
        return str(namespace).rstrip(":")
    return f"memory:{config.scope}"


def _redis_vector_index_name(config: StoreConfig) -> Optional[str]:
    options = config.options or {}
    vector_name = options.get("vector_index") or options.get("redis_vector_index")
    if vector_name:
        return str(vector_name)
    return None


def ensure_redis_ready(
    config: StoreConfig,
    *,
    vector_dimensions: Optional[int] = None,
) -> StepResult:
    """Ping Redis and provision RediSearch indexes when required."""

    scope = config.scope
    if redis is None:
        return StepResult(
            name=f"redis:{scope}",
            ok=False,
            detail="redis-py is not installed",
            hint="pip install redis>=5.0 and rerun the setup script",
        )

    settings: RedisSettings = config.redis or RedisSettings()

    try:
        if settings.url:
            client = redis.Redis.from_url(settings.url, decode_responses=False)  # type: ignore[attr-defined]
            connection_repr = settings.url
        else:
            kwargs = dict(settings.options)
            if settings.username:
                kwargs.setdefault("username", settings.username)
            if settings.password:
                kwargs.setdefault("password", settings.password)
            if settings.ssl is not None:
                kwargs.setdefault("ssl", settings.ssl)
            client = redis.Redis(  # type: ignore[call-arg]
                host=settings.host,
                port=settings.port,
                db=settings.db,
                **kwargs,
            )
            connection_repr = f"{settings.host}:{settings.port}/{settings.db}"
    except Exception as exc:  # pragma: no cover - redis misconfiguration
        return StepResult(
            name=f"redis:{scope}",
            ok=False,
            detail=f"failed to construct Redis client: {exc}",
            hint="verify MEMORY_*_REDIS_* environment variables",
        )

    try:
        client.ping()  # type: ignore[attr-defined]
    except RedisError as exc:
        return StepResult(
            name=f"redis:{scope}",
            ok=False,
            detail=f"unable to ping Redis at {connection_repr}: {exc}",
            hint="ensure the Redis service is running and accessible",
        )

    namespace = _redis_namespace(config)
    vector_index = _redis_vector_index_name(config)
    options = config.options or {}
    dimensions = vector_dimensions or _coerce_int(options, "vector_dimensions", _coerce_int(options, "embedding_dimensions", 1536))

    index_created = False
    if vector_index:
        if not hasattr(client, "ft"):
            return StepResult(
                name=f"redis:{scope}",
                ok=False,
                detail="RedisSearch module is unavailable on the target server",
                hint="install RediSearch or remove the vector_index option",
            )
        if any(field is None for field in (IndexDefinition, IndexType, NumericField, TagField, TextField, VectorField)):
            return StepResult(
                name=f"redis:{scope}",
                ok=False,
                detail="redisearch Python helpers are missing",
                hint="upgrade redis-py to 5.0+ with search extras enabled",
            )
        search = client.ft(vector_index)  # type: ignore[attr-defined]
        try:
            search.info()
        except ResponseError as exc:  # type: no cover - index absent
            if "Unknown Index name" not in str(exc):
                return StepResult(
                    name=f"redis:{scope}",
                    ok=False,
                    detail=f"unable to inspect vector index '{vector_index}': {exc}",
                    hint="verify the RediSearch module is loaded",
                )
            try:
                schema = [
                    TextField("content"),
                    TagField("tags"),
                    TagField("session_id"),
                    TagField("agent_id"),
                    NumericField("score"),
                ]
                if dimensions:
                    schema.append(
                        VectorField(
                            "embedding_blob",
                            "FLAT",
                            {
                                "TYPE": "FLOAT32",
                                "DIM": dimensions,
                                "DISTANCE_METRIC": "L2",
                                "INITIAL_CAP": 1000,
                            },
                        )
                    )
                definition = IndexDefinition(prefix=[f"{namespace}:record:"], index_type=IndexType.HASH)
                search.create_index(schema, definition=definition)
                index_created = True
            except ResponseError as creation_error:
                return StepResult(
                    name=f"redis:{scope}",
                    ok=False,
                    detail=f"failed to create vector index '{vector_index}': {creation_error}",
                    hint="check RedisSearch permissions and configuration",
                )
        except Exception as exc:  # pragma: no cover - unexpected redis errors
            return StepResult(
                name=f"redis:{scope}",
                ok=False,
                detail=f"failed to query vector index '{vector_index}': {exc}",
                hint="inspect Redis logs for RediSearch errors",
            )

    detail = f"ping successful for {connection_repr}"
    if vector_index:
        if index_created:
            detail += f"; created RediSearch index '{vector_index}'"
        else:
            detail += f"; vector index '{vector_index}' is available"
    return StepResult(name=f"redis:{scope}", ok=True, detail=detail)


def _build_pg_conninfo(settings: PostgresSettings) -> str:
    if settings.dsn:
        return settings.dsn
    parts = []
    if settings.host:
        parts.append(f"host={settings.host}")
    if settings.port:
        parts.append(f"port={settings.port}")
    if settings.database:
        parts.append(f"dbname={settings.database}")
    if settings.user:
        parts.append(f"user={settings.user}")
    if settings.password:
        parts.append(f"password={settings.password}")
    if settings.sslmode:
        parts.append(f"sslmode={settings.sslmode}")
    for key, value in (settings.options or {}).items():
        parts.append(f"{key}={value}")
    return " ".join(parts)


def ensure_postgres_ready(
    config: StoreConfig,
    *,
    migrations_path: Path = DEFAULT_MIGRATIONS_DIR,
) -> StepResult:
    """Run SQL migrations and validate pgvector availability."""

    scope = config.scope
    if psycopg is None:
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail="psycopg (v3) is not installed",
            hint="pip install psycopg[binary,pool]>=3.1",
        )

    settings: PostgresSettings = config.postgres or PostgresSettings()
    conninfo = _build_pg_conninfo(settings)

    connect_kwargs = dict(settings.options)
    try:
        conn = psycopg.connect(conninfo, **connect_kwargs)  # type: ignore[call-arg]
    except Exception as exc:
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail=f"unable to connect using '{conninfo or settings.dsn}': {exc}",
            hint="confirm credentials and network access to PostgreSQL",
        )

    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception as exc:
        conn.close()
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail=f"connection check failed: {exc}",
            hint="review PostgreSQL logs for authentication errors",
        )

    migration_files = sorted(path for path in Path(migrations_path).glob("*.sql") if path.is_file())
    if not migration_files:
        conn.close()
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail=f"no SQL migrations found in {migrations_path}",
            hint="verify the repository checkout includes internal/db/migrations",
        )

    try:
        with conn.cursor() as cur:
            for migration in migration_files:
                sql_text = migration.read_text(encoding="utf-8")
                if not sql_text.strip():
                    continue
                try:
                    cur.execute(sql_text)
                except DuplicateObject:
                    continue
    except psycopg.errors.UndefinedFile as exc:  # type: ignore[attr-defined]
        conn.close()
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail=f"required extension is missing: {exc}",
            hint="install the pgvector and uuid-ossp extensions on the target database",
        )
    except Exception as exc:
        conn.close()
        return StepResult(
            name=f"postgres:{scope}",
            ok=False,
            detail=f"migration failed: {exc}",
            hint="rerun with --migrations-dir to point at valid SQL scripts",
        )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.memory_entries')")
            table_name = cur.fetchone()[0]
            if table_name is None:
                raise RuntimeError("memory_entries table was not created")
            cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp')")
            installed_extensions = {row[0] for row in cur.fetchall()}
            missing = {"vector", "uuid-ossp"} - installed_extensions
            if missing:
                return StepResult(
                    name=f"postgres:{scope}",
                    ok=False,
                    detail=f"database missing extensions: {', '.join(sorted(missing))}",
                    hint="install required extensions (CREATE EXTENSION vector; CREATE EXTENSION \"uuid-ossp\";)",
                )
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'memory_entries' AND indexname = 'idx_memory_entries_embedding'"
            )
            has_vector_index = cur.fetchone() is not None
    finally:
        conn.close()

    detail = f"connected using '{conninfo or settings.dsn}'"
    if has_vector_index:
        detail += "; pgvector index ready"
    else:
        detail += "; warning: vector index not found"
    hint = None if has_vector_index else "re-run migrations or create the ivfflat index manually"
    return StepResult(name=f"postgres:{scope}", ok=has_vector_index, detail=detail, hint=hint)


def _iter_configs(config: StoreConfig, *, kind: str) -> Iterable[StoreConfig]:
    if config.backend.lower() == kind:
        yield config


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional path to a memory config JSON override")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=DEFAULT_MIGRATIONS_DIR,
        help="Directory containing SQL migrations for PostgreSQL backends",
    )
    parser.add_argument(
        "--redis-only",
        action="store_true",
        help="Only prepare Redis backends (skip PostgreSQL steps)",
    )
    parser.add_argument(
        "--postgres-only",
        action="store_true",
        help="Only prepare PostgreSQL backends (skip Redis steps)",
    )
    args = parser.parse_args(argv)

    if args.redis_only and args.postgres_only:
        parser.error("--redis-only and --postgres-only are mutually exclusive")

    config = load_memory_configuration(args.config)

    results: list[StepResult] = []
    if not args.postgres_only:
        for store in (
            config.short_term.copy(),
            config.long_term.copy(),
            config.combined.copy(),
        ):
            for redis_config in _iter_configs(store, kind="redis"):
                results.append(ensure_redis_ready(redis_config))

    if not args.redis_only:
        for store in (
            config.short_term.copy(),
            config.long_term.copy(),
            config.combined.copy(),
        ):
            for pg_config in _iter_configs(store, kind="postgres"):
                results.append(ensure_postgres_ready(pg_config, migrations_path=args.migrations_dir))

    if not results:
        print("No Redis or PostgreSQL backends configured; nothing to do.")
        return 0

    all_ok = True
    for result in results:
        print(result.format())
        all_ok = all_ok and result.ok

    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=os.getenv("SETUP_MEMORY_LOG", "INFO"))
    sys.exit(main())
