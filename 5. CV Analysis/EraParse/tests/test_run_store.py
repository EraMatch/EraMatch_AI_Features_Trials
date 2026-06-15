from datetime import UTC, datetime
from pathlib import Path

import duckdb

from eraparse.evaluate import evaluate_document
from eraparse.models import RunProvenance, RunRecord
from eraparse.run_store import (
    initialize_database,
    insert_artifact,
    insert_run,
    insert_sample_result,
)


def test_run_store_initializes_and_writes(
    tmp_path: Path, reduced_target: dict[str, object]
) -> None:
    database = tmp_path / "runs.duckdb"
    initialize_database(database)
    run = RunRecord(
        run_id="run-1",
        kind="evaluation",
        status="completed",
        provenance=RunProvenance(seed=20260609, manifest_hash="abc"),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    insert_run(database, run)
    result = evaluate_document(reduced_target, reduced_target)
    insert_sample_result(
        database, run_id=run.run_id, cv_id="cv_00001", split="debug_50", result=result
    )
    insert_artifact(database, run_id=run.run_id, kind="prediction", artifact_path="prediction.json")

    with duckdb.connect(str(database)) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM sample_results").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM field_results").fetchone() == (
            len(result.field_results),
        )
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone() == (1,)
