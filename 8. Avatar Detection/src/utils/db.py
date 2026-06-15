"""SQLite experiment logger for avatar detection trials.

Three tables: experiments, training_history, dataset_audit.
Uses WAL mode for safe concurrent access on Modal volumes.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import RESULTS_ROOT

DB_PATH = RESULTS_ROOT / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_name            TEXT NOT NULL,
    model_arch            TEXT NOT NULL,
    backbone              TEXT,
    train_datasets        TEXT,
    test_dataset          TEXT,
    n_train               INTEGER,
    n_val                 INTEGER,
    n_test                INTEGER,
    use_srm_branch        INTEGER DEFAULT 1,
    use_supcon            INTEGER DEFAULT 1,
    use_freq_augmentation INTEGER DEFAULT 1,
    use_attention_gate    INTEGER DEFAULT 1,
    test_auc              REAL,
    test_f1               REAL,
    test_accuracy         REAL,
    test_precision        REAL,
    test_recall           REAL,
    test_specificity      REAL,
    test_eer              REAL,
    test_tpr_at_fpr1      REAL,
    val_best_auc          REAL,
    epochs                INTEGER,
    batch_size            INTEGER,
    learning_rate         REAL,
    weight_decay          REAL,
    supcon_weight         REAL,
    checkpoint_path       TEXT,
    plots_dir             TEXT,
    config_json           TEXT,
    gpu_type              TEXT,
    training_time_s       INTEGER,
    timestamp             TEXT,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS training_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL,
    epoch           INTEGER NOT NULL,
    train_loss      REAL,
    val_loss        REAL,
    val_auc         REAL,
    val_f1          REAL,
    lr              REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS dataset_audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name    TEXT,
    n_total         INTEGER,
    n_real          INTEGER,
    n_fake          INTEGER,
    face_rate       REAL,
    avg_resolution  TEXT,
    generators      TEXT,
    notes           TEXT,
    audited_at      TEXT
);
"""


def _connect(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    conn = _connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def log_experiment(
    db_path: Optional[Union[str, Path]] = None,
    trial_name: str = "",
    model_arch: str = "",
    backbone: Optional[str] = None,
    train_datasets: Optional[str] = None,
    test_dataset: Optional[str] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    use_srm_branch: int = 1,
    use_supcon: int = 1,
    use_freq_augmentation: int = 1,
    use_attention_gate: int = 1,
    test_auc: Optional[float] = None,
    test_f1: Optional[float] = None,
    test_accuracy: Optional[float] = None,
    test_precision: Optional[float] = None,
    test_recall: Optional[float] = None,
    test_specificity: Optional[float] = None,
    test_eer: Optional[float] = None,
    test_tpr_at_fpr1: Optional[float] = None,
    val_best_auc: Optional[float] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    supcon_weight: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    plots_dir: Optional[str] = None,
    config_json: Optional[str] = None,
    gpu_type: Optional[str] = None,
    training_time_s: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    conn = _connect(db_path)
    cursor = conn.execute(
        """INSERT INTO experiments (
            trial_name, model_arch, backbone, train_datasets, test_dataset,
            n_train, n_val, n_test,
            use_srm_branch, use_supcon, use_freq_augmentation, use_attention_gate,
            test_auc, test_f1, test_accuracy, test_precision, test_recall,
            test_specificity, test_eer, test_tpr_at_fpr1, val_best_auc,
            epochs, batch_size, learning_rate, weight_decay, supcon_weight,
            checkpoint_path, plots_dir, config_json,
            gpu_type, training_time_s, timestamp, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trial_name,
            model_arch,
            backbone,
            train_datasets,
            test_dataset,
            n_train,
            n_val,
            n_test,
            use_srm_branch,
            use_supcon,
            use_freq_augmentation,
            use_attention_gate,
            test_auc,
            test_f1,
            test_accuracy,
            test_precision,
            test_recall,
            test_specificity,
            test_eer,
            test_tpr_at_fpr1,
            val_best_auc,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
            supcon_weight,
            checkpoint_path,
            plots_dir,
            config_json,
            gpu_type,
            training_time_s,
            datetime.utcnow().isoformat(),
            notes,
        ),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    exp_id = cursor.lastrowid
    conn.close()
    return exp_id


def log_training_history(
    db_path: Optional[Union[str, Path]] = None,
    experiment_id: int = 0,
    epoch: int = 0,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    val_auc: Optional[float] = None,
    val_f1: Optional[float] = None,
    lr: Optional[float] = None,
) -> int:
    conn = _connect(db_path)
    cursor = conn.execute(
        """INSERT INTO training_history (
            experiment_id, epoch, train_loss, val_loss, val_auc, val_f1, lr
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, epoch, train_loss, val_loss, val_auc, val_f1, lr),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def log_dataset_audit(
    db_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    n_total: Optional[int] = None,
    n_real: Optional[int] = None,
    n_fake: Optional[int] = None,
    face_rate: Optional[float] = None,
    avg_resolution: Optional[str] = None,
    generators: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    conn = _connect(db_path)
    cursor = conn.execute(
        """INSERT INTO dataset_audit (
            dataset_name, n_total, n_real, n_fake, face_rate,
            avg_resolution, generators, notes, audited_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            dataset_name,
            n_total,
            n_real,
            n_fake,
            face_rate,
            avg_resolution,
            generators,
            notes,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_experiments(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    result = [dict(row) for row in rows]
    conn.close()
    return result


def get_experiment_by_name(
    trial_name: str, db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM experiments WHERE trial_name = ? ORDER BY id",
        (trial_name,),
    ).fetchall()
    result = [dict(row) for row in rows]
    conn.close()
    return result
