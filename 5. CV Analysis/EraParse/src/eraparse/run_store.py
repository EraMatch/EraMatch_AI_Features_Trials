import json
from pathlib import Path
from typing import Any

import duckdb

from eraparse.models import DocumentEvaluation, RunRecord

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    code_revision VARCHAR,
    manifest_hash VARCHAR,
    environment_json JSON NOT NULL,
    seed BIGINT NOT NULL,
    resolved_config_json JSON NOT NULL,
    model_id VARCHAR,
    parser_id VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error VARCHAR
);

CREATE TABLE IF NOT EXISTS sample_results (
    run_id VARCHAR NOT NULL,
    cv_id VARCHAR NOT NULL,
    split VARCHAR NOT NULL,
    json_valid BOOLEAN NOT NULL,
    schema_valid BOOLEAN NOT NULL,
    micro_score DOUBLE NOT NULL,
    macro_score DOUBLE NOT NULL,
    unsupported_evidence_rate DOUBLE NOT NULL,
    result_json JSON NOT NULL,
    PRIMARY KEY (run_id, cv_id)
);

CREATE TABLE IF NOT EXISTS field_results (
    run_id VARCHAR NOT NULL,
    cv_id VARCHAR NOT NULL,
    field_path VARCHAR NOT NULL,
    metric VARCHAR NOT NULL,
    score DOUBLE NOT NULL,
    supported BOOLEAN,
    result_json JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    sha256 VARCHAR,
    metadata_json JSON NOT NULL
);
"""


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as connection:
        connection.execute(SCHEMA_SQL)


def insert_run(path: Path, run: RunRecord) -> None:
    initialize_database(path)
    provenance = run.provenance
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.run_id,
                run.kind,
                run.status,
                provenance.code_revision,
                provenance.manifest_hash,
                json.dumps(provenance.environment),
                provenance.seed,
                json.dumps(provenance.resolved_config),
                run.model_id,
                run.parser_id,
                run.started_at,
                run.completed_at,
                run.error,
            ],
        )


def insert_sample_result(
    path: Path,
    *,
    run_id: str,
    cv_id: str,
    split: str,
    result: DocumentEvaluation,
) -> None:
    initialize_database(path)
    result_json = json.dumps(result.model_dump(mode="json"))
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO sample_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                cv_id,
                split,
                result.json_valid,
                result.schema_valid,
                result.micro_score,
                result.macro_score,
                result.unsupported_evidence_rate,
                result_json,
            ],
        )
        connection.execute(
            "DELETE FROM field_results WHERE run_id = ? AND cv_id = ?", [run_id, cv_id]
        )
        connection.executemany(
            "INSERT INTO field_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                [
                    run_id,
                    cv_id,
                    field.path,
                    field.metric,
                    field.score,
                    field.supported,
                    json.dumps(field.model_dump(mode="json")),
                ]
                for field in result.field_results
            ],
        )


def insert_artifact(
    path: Path,
    *,
    run_id: str,
    kind: str,
    artifact_path: str,
    sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    initialize_database(path)
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
            [run_id, kind, artifact_path, sha256, json.dumps(metadata or {})],
        )
