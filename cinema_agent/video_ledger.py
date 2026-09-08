import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LedgerAdmission:
    allowed: bool
    reason: str | None = None
    existing_job: dict[str, Any] | None = None


@dataclass(frozen=True)
class AllowanceSnapshot:
    global_limit: int
    global_used: int
    global_remaining: int
    per_ip_limit: int
    per_ip_used: int
    per_ip_remaining: int
    director_cut_global_limit: int
    director_cut_global_used: int
    director_cut_global_remaining: int
    authorized_replacement_limit: int
    authorized_replacement_used: int
    authorized_replacement_remaining: int
    authorized_replacement_for_job_id: str | None = None


class VideoLedger:
    """Durable real-generation accounting and job metadata."""

    def __init__(
        self,
        path: str | None = None,
        *,
        global_limit: int = 1,
        per_ip_limit: int = 1,
        director_cut_global_limit: int = 0,
    ):
        self.path = path or os.getenv("VIDEO_GENERATION_DB_PATH", "data/video_jobs.sqlite3")
        self.global_limit = max(0, global_limit)
        self.per_ip_limit = max(0, per_ip_limit)
        self.director_cut_global_limit = max(0, director_cut_global_limit)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        if self.path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS video_generation_submissions (
                    job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    client_ip TEXT,
                    kind TEXT NOT NULL,
                    allowance_source TEXT NOT NULL DEFAULT 'normal'
                        CHECK (allowance_source IN ('normal', 'authorized_replacement')),
                    replacement_for_job_id TEXT,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'unknown', 'accepted', 'released')),
                    accepted_at REAL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_video_submissions_client
                    ON video_generation_submissions(client_id, state);

                CREATE TABLE IF NOT EXISTS durable_video_jobs (
                    job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    client_ip TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scene_key TEXT NOT NULL,
                    source_signature TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    operation_id TEXT,
                     model TEXT,
                    request_fingerprint TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    revision_approved INTEGER NOT NULL DEFAULT 0,
                    output_json TEXT,
                    video_path TEXT,
                    critique_context_json TEXT,
                    storyboard_bytes BLOB,
                    storyboard_mime_type TEXT,
                    replacement_for_job_id TEXT,
                    output_reference TEXT,
                    error_code TEXT,
                     error_message TEXT,
                     provider_error_code INTEGER,
                     provider_error_status TEXT,
                     provider_error_category TEXT,
                     provider_error_message TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_durable_video_cache
                    ON durable_video_jobs(client_id, cache_key);
                CREATE TABLE IF NOT EXISTS video_replacement_authorizations (
                    original_job_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    authorized_at REAL NOT NULL,
                    authorized_by TEXT NOT NULL,
                    authorization_reason TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'available'
                        CHECK (state IN ('available', 'reserved', 'consumed')),
                    reserved_job_id TEXT,
                    reserved_at REAL,
                    consumed_at REAL,
                    released_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_video_replacement_client
                    ON video_replacement_authorizations(client_id, state);
                CREATE INDEX IF NOT EXISTS idx_durable_video_active
                    ON durable_video_jobs(client_id, status);
                """
            )
            for column, definition in (
                ("client_ip", "TEXT"),
                ("critique_context_json", "TEXT"),
                ("storyboard_bytes", "BLOB"),
                ("storyboard_mime_type", "TEXT"),
                ("replacement_for_job_id", "TEXT"),
                ("output_reference", "TEXT"),
                ("model", "TEXT"),
                ("provider_error_code", "INTEGER"),
                ("provider_error_status", "TEXT"),
                ("provider_error_category", "TEXT"),
                ("provider_error_message", "TEXT"),
            ):
                try:
                    connection.execute(
                        f"ALTER TABLE durable_video_jobs ADD COLUMN {column} {definition}"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            connection.execute(
                "UPDATE durable_video_jobs SET client_ip = client_id WHERE client_ip IS NULL OR client_ip = ''"
            )
            try:
                connection.execute(
                    "ALTER TABLE video_generation_submissions ADD COLUMN client_ip TEXT"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            for column, definition in (
                ("allowance_source", "TEXT NOT NULL DEFAULT 'normal'"),
                ("replacement_for_job_id", "TEXT"),
            ):
                try:
                    connection.execute(
                        f"ALTER TABLE video_generation_submissions ADD COLUMN {column} {definition}"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" not in str(exc).lower():
                        raise
            submission_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'video_generation_submissions'"
            ).fetchone()[0]
            if "unknown" not in submission_sql:
                connection.execute("DROP INDEX IF EXISTS idx_video_submissions_client")
                connection.execute(
                    "ALTER TABLE video_generation_submissions RENAME TO video_generation_submissions_legacy"
                )
                connection.execute(
                    """
                    CREATE TABLE video_generation_submissions (
                        job_id TEXT PRIMARY KEY,
                        client_id TEXT NOT NULL,
                        client_ip TEXT,
                        kind TEXT NOT NULL,
                        allowance_source TEXT NOT NULL DEFAULT 'normal'
                            CHECK (allowance_source IN ('normal', 'authorized_replacement')),
                        replacement_for_job_id TEXT,
                        state TEXT NOT NULL CHECK (state IN ('pending', 'unknown', 'accepted', 'released')),
                        accepted_at REAL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO video_generation_submissions
                        (job_id, client_id, client_ip, kind, allowance_source,
                         replacement_for_job_id, state, accepted_at, created_at)
                    SELECT job_id, client_id, client_id, kind, 'normal',
                           NULL, state, accepted_at, created_at
                    FROM video_generation_submissions_legacy
                    """
                )
                connection.execute("DROP TABLE video_generation_submissions_legacy")
                connection.execute(
                    """
                    CREATE INDEX idx_video_submissions_client
                        ON video_generation_submissions(client_id, state)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX idx_video_submissions_ip
                        ON video_generation_submissions(client_ip, state)
                    """
                )
            connection.execute(
                "UPDATE video_generation_submissions SET client_ip = client_id WHERE client_ip IS NULL OR client_ip = ''"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_video_submissions_ip ON video_generation_submissions(client_ip, state)"
            )
            connection.execute("DROP INDEX IF EXISTS idx_durable_video_scene_kind")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_durable_video_scene_kind
                    ON durable_video_jobs(client_id, scene_key, kind)
                    WHERE replacement_for_job_id IS NULL
                """
            )

    def cached_job(self, client_id: str, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM durable_video_jobs WHERE client_id = ? AND cache_key = ?",
                (client_id, cache_key),
            ).fetchone()
            return dict(row) if row else None

    def latest_completed_job(self, client_id: str, *, kind: str = "first_cut") -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM durable_video_jobs
                WHERE client_id = ? AND kind = ? AND status = 'completed'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (client_id, kind),
            ).fetchone()
            return dict(row) if row else None

    def authorize_replacement(
        self,
        original_job_id: str,
        *,
        authorized_by: str,
        authorization_reason: str,
        authorized_at: float,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM video_replacement_authorizations WHERE original_job_id = ?",
                (original_job_id,),
            ).fetchone()
            if existing:
                connection.execute("COMMIT")
                return dict(existing)
            original = connection.execute(
                """
                SELECT j.client_id, j.kind, j.status, s.state
                FROM durable_video_jobs AS j
                JOIN video_generation_submissions AS s ON s.job_id = j.job_id
                WHERE j.job_id = ?
                """,
                (original_job_id,),
            ).fetchone()
            if (
                not original
                or original["kind"] != "first_cut"
                or original["status"] != "failed"
                or original["state"] != "accepted"
            ):
                connection.execute("ROLLBACK")
                raise ValueError("Replacement authorization requires an accepted failed First Cut.")
            connection.execute(
                """
                INSERT INTO video_replacement_authorizations
                    (original_job_id, client_id, authorized_at, authorized_by,
                     authorization_reason, state)
                VALUES (?, ?, ?, ?, ?, 'available')
                """,
                (
                    original_job_id,
                    original["client_id"],
                    authorized_at,
                    authorized_by,
                    authorization_reason,
                ),
            )
            connection.execute("COMMIT")
            row = connection.execute(
                "SELECT * FROM video_replacement_authorizations WHERE original_job_id = ?",
                (original_job_id,),
            ).fetchone()
            return dict(row)

    def replacement_authorization(
        self,
        original_job_id: str,
        client_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.*, j.kind AS original_kind, j.status AS original_status,
                       s.state AS original_submission_state
                FROM video_replacement_authorizations AS a
                JOIN durable_video_jobs AS j ON j.job_id = a.original_job_id
                JOIN video_generation_submissions AS s ON s.job_id = a.original_job_id
                WHERE a.original_job_id = ? AND a.client_id = ?
                """,
                (original_job_id, client_id),
            ).fetchone()
            return dict(row) if row else None

    def validate_replacement_authorization(
        self,
        original_job_id: str,
        client_id: str,
        *,
        require_available: bool,
    ) -> dict[str, Any]:
        row = self.replacement_authorization(original_job_id, client_id)
        if (
            not row
            or row["original_kind"] != "first_cut"
            or row["original_status"] != "failed"
            or row["original_submission_state"] != "accepted"
            or (require_available and row["state"] != "available")
        ):
            raise ValueError("The authorized replacement is unavailable for this client or job.")
        return row

    def begin_submission(
        self,
        *,
        job_id: str,
        client_id: str,
        client_ip: str | None = None,
        kind: str,
        scene_key: str,
        source_signature: str,
        provider: str,
        created_at: float,
        request_fingerprint: str,
        cache_key: str,
        model: str | None = None,
        critique_context_json: str | None = None,
        storyboard_bytes: bytes | None = None,
        storyboard_mime_type: str | None = None,
        replacement_for_job_id: str | None = None,
    ) -> LedgerAdmission:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM durable_video_jobs WHERE client_id = ? AND cache_key = ?",
                (client_id, cache_key),
            ).fetchone()
            if existing:
                if existing["request_fingerprint"] != request_fingerprint:
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "duplicate_scene_kind", dict(existing))
                connection.execute("COMMIT")
                return LedgerAdmission(False, "cached", dict(existing))

            scene_existing = connection.execute(
                """
                SELECT * FROM durable_video_jobs
                WHERE client_id = ? AND scene_key = ? AND kind = ?
                  AND replacement_for_job_id IS NULL
                """,
                (client_id, scene_key, kind),
            ).fetchone()
            if scene_existing and replacement_for_job_id is None:
                connection.execute("COMMIT")
                return LedgerAdmission(False, "duplicate_scene_kind", dict(scene_existing))

            active_existing = connection.execute(
                """
                SELECT * FROM durable_video_jobs
                WHERE client_id = ? AND status IN ('submitting', 'submission_unknown', 'queued', 'generating')
                LIMIT 1
                """,
                (client_id,),
            ).fetchone()
            if active_existing:
                connection.execute("COMMIT")
                return LedgerAdmission(False, "active_job", dict(active_existing))

            if replacement_for_job_id is None:
                used_global = self._used_count(connection)
                used_ip = self._used_count(connection, client_ip=client_ip or client_id)
                if used_global >= self.global_limit:
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "global_limit")
                if used_ip >= self.per_ip_limit:
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "per_ip_limit")
                if kind == "director_cut" and self._used_count(connection, kind=kind) >= self.director_cut_global_limit:
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "director_cut_global_limit")
            else:
                replacement = connection.execute(
                    """
                    SELECT a.*, j.kind AS original_kind, j.status AS original_status,
                           s.state AS original_submission_state
                    FROM video_replacement_authorizations AS a
                    JOIN durable_video_jobs AS j ON j.job_id = a.original_job_id
                    JOIN video_generation_submissions AS s ON s.job_id = a.original_job_id
                    WHERE a.original_job_id = ? AND a.client_id = ?
                    """,
                    (replacement_for_job_id, client_id),
                ).fetchone()
                if (
                    not replacement
                    or replacement["original_kind"] != "first_cut"
                    or replacement["original_status"] != "failed"
                    or replacement["original_submission_state"] != "accepted"
                    or replacement["state"] != "available"
                ):
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "replacement_unavailable")
                reserved = connection.execute(
                    """
                    UPDATE video_replacement_authorizations
                    SET state = 'reserved', reserved_job_id = ?, reserved_at = ?,
                        released_at = NULL
                    WHERE original_job_id = ? AND client_id = ? AND state = 'available'
                    """,
                    (job_id, created_at, replacement_for_job_id, client_id),
                )
                if reserved.rowcount != 1:
                    connection.execute("COMMIT")
                    return LedgerAdmission(False, "replacement_unavailable")

            connection.execute(
                """
                INSERT INTO video_generation_submissions
                    (job_id, client_id, client_ip, kind, allowance_source,
                     replacement_for_job_id, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    job_id,
                    client_id,
                    client_ip or client_id,
                    kind,
                    "authorized_replacement" if replacement_for_job_id else "normal",
                    replacement_for_job_id,
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO durable_video_jobs
                    (job_id, client_id, client_ip, kind, status, scene_key, source_signature, provider,
                     created_at, updated_at, model, request_fingerprint, cache_key,
                     critique_context_json, storyboard_bytes, storyboard_mime_type,
                     replacement_for_job_id)
                VALUES (?, ?, ?, ?, 'submitting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    client_id,
                    client_ip or client_id,
                    kind,
                    scene_key,
                    source_signature,
                    provider,
                    created_at,
                    created_at,
                    model,
                    request_fingerprint,
                    cache_key,
                    critique_context_json,
                    storyboard_bytes,
                    storyboard_mime_type,
                    replacement_for_job_id,
                ),
            )
            connection.execute("COMMIT")
            return LedgerAdmission(True)

    def accept_submission(self, job_id: str, operation_id: str, updated_at: float) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE video_generation_submissions
                SET state = 'accepted', accepted_at = ?
                WHERE job_id = ? AND state IN ('pending', 'unknown')
                """,
                (updated_at, job_id),
            )
            if updated.rowcount:
                replacement = connection.execute(
                    """
                    SELECT replacement_for_job_id
                    FROM video_generation_submissions
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if replacement and replacement["replacement_for_job_id"]:
                    connection.execute(
                        """
                        UPDATE video_replacement_authorizations
                        SET state = 'consumed', consumed_at = ?
                        WHERE original_job_id = ? AND state = 'reserved'
                          AND reserved_job_id = ?
                        """,
                        (updated_at, replacement["replacement_for_job_id"], job_id),
                    )
            connection.execute(
                """
                UPDATE durable_video_jobs
                SET status = 'queued', operation_id = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (operation_id, updated_at, job_id),
            )
            connection.execute("COMMIT")

    def mark_submission_unknown(
        self,
        job_id: str,
        operation_id: str | None,
        error_code: str,
        error_message: str,
        updated_at: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE video_generation_submissions
                SET state = 'unknown'
                WHERE job_id = ? AND state = 'pending'
                """,
                (job_id,),
            )
            connection.execute(
                """
                UPDATE durable_video_jobs
                SET status = 'submission_unknown', operation_id = ?,
                    error_code = ?, error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (operation_id, error_code, error_message, updated_at, job_id),
            )
            connection.execute("COMMIT")

    def release_submission(self, job_id: str, error_code: str, error_message: str, updated_at: float) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            released = connection.execute(
                """
                UPDATE video_generation_submissions
                SET state = 'released'
                WHERE job_id = ? AND state = 'pending'
                """,
                (job_id,),
            )
            if released.rowcount:
                replacement = connection.execute(
                    """
                    SELECT replacement_for_job_id
                    FROM video_generation_submissions
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                if replacement and replacement["replacement_for_job_id"]:
                    connection.execute(
                        """
                        UPDATE video_replacement_authorizations
                        SET state = 'available', reserved_job_id = NULL,
                            reserved_at = NULL, released_at = ?
                        WHERE original_job_id = ? AND state = 'reserved'
                          AND reserved_job_id = ?
                        """,
                        (updated_at, replacement["replacement_for_job_id"], job_id),
                    )
            connection.execute(
                """
                UPDATE durable_video_jobs
                SET status = 'failed', error_code = ?, error_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (error_code, error_message, updated_at, job_id),
            )
            connection.execute("COMMIT")

    def resume_accepted_operation(self, job_id: str, updated_at: float) -> None:
        """Reopen a locally timed-out accepted operation without changing accounting."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE durable_video_jobs
                SET status = 'generating',
                    error_code = NULL,
                    error_message = NULL,
                    provider_error_code = NULL,
                    provider_error_status = NULL,
                    provider_error_category = NULL,
                    provider_error_message = NULL,
                    updated_at = ?
                WHERE job_id = ?
                  AND status = 'failed'
                  AND error_code = 'generation_timeout'
                  AND operation_id IS NOT NULL
                """,
                (updated_at, job_id),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        updated_at: float,
        revision_approved: bool,
        output: dict[str, Any] | None,
        video_path: str | None,
        output_reference: str | None,
        error_code: str | None,
        error_message: str | None,
        provider_error_code: int | None,
        provider_error_status: str | None,
        provider_error_category: str | None,
        provider_error_message: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE durable_video_jobs
                SET status = ?, updated_at = ?, revision_approved = ?,
                    output_json = ?, video_path = ?, output_reference = ?,
                     error_code = ?, error_message = ?,
                     provider_error_code = ?, provider_error_status = ?,
                     provider_error_category = ?, provider_error_message = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    updated_at,
                    int(revision_approved),
                    json.dumps(output, separators=(",", ":")) if output else None,
                    video_path,
                    output_reference,
                    error_code,
                    error_message,
                    provider_error_code,
                    provider_error_status,
                    provider_error_category,
                    provider_error_message,
                    job_id,
                ),
            )

    def load_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM durable_video_jobs WHERE provider = 'vertex'"
            ).fetchall()
            return [dict(row) for row in rows]

    def allowance(self, client_id: str, client_ip: str | None = None) -> AllowanceSnapshot:
        with self._connect() as connection:
            global_used = self._used_count(connection)
            ip_used = self._used_count(connection, client_ip=client_ip or client_id)
            director_used = self._used_count(connection, kind="director_cut")
            replacement_rows = connection.execute(
                """
                SELECT a.original_job_id, a.state
                FROM video_replacement_authorizations AS a
                JOIN durable_video_jobs AS j ON j.job_id = a.original_job_id
                WHERE a.client_id = ? AND j.kind = 'first_cut'
                ORDER BY authorized_at ASC
                """,
                (client_id,),
            ).fetchall()
            replacement_limit = len(replacement_rows)
            replacement_used = sum(row["state"] == "consumed" for row in replacement_rows)
            replacement_remaining = sum(row["state"] == "available" for row in replacement_rows)
            replacement_for_job_id = (
                next(
                    (
                        row["original_job_id"]
                        for row in replacement_rows
                        if row["state"] == "available"
                    ),
                    None,
                )
            )
        return AllowanceSnapshot(
            global_limit=self.global_limit,
            global_used=global_used,
            global_remaining=max(0, self.global_limit - global_used),
            per_ip_limit=self.per_ip_limit,
            per_ip_used=ip_used,
            per_ip_remaining=max(0, self.per_ip_limit - ip_used),
            director_cut_global_limit=self.director_cut_global_limit,
            director_cut_global_used=director_used,
            director_cut_global_remaining=max(0, self.director_cut_global_limit - director_used),
            authorized_replacement_limit=replacement_limit,
            authorized_replacement_used=replacement_used,
            authorized_replacement_remaining=replacement_remaining,
            authorized_replacement_for_job_id=replacement_for_job_id,
        )

    @staticmethod
    def _used_count(
        connection: sqlite3.Connection,
        *,
        client_id: str | None = None,
        client_ip: str | None = None,
        kind: str | None = None,
    ) -> int:
        clauses = [
            "state IN ('pending', 'unknown', 'accepted')",
            "allowance_source = 'normal'",
        ]
        values: list[str] = []
        if client_id is not None:
            clauses.append("client_id = ?")
            values.append(client_id)
        if client_ip is not None:
            clauses.append("client_ip = ?")
            values.append(client_ip)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        row = connection.execute(
            f"SELECT COUNT(*) AS count FROM video_generation_submissions WHERE {' AND '.join(clauses)}",
            values,
        ).fetchone()
        return int(row["count"])