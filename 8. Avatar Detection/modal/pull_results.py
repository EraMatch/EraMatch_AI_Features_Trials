"""Pull all results from avatar-results-vol to local disk."""
from pathlib import Path
import modal

RESULTS_ROOT = Path("/results")
results_vol  = modal.Volume.from_name("avatar-results-vol", create_if_missing=False)
app = modal.App("avatar-pull-results")

image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas==2.2.3")

@app.function(
    image=image,
    memory=2048,
    timeout=5 * 60,
    volumes={str(RESULTS_ROOT): results_vol},
)
def list_and_read() -> dict:
    import os, json, sqlite3

    out = {}

    # ── ablation_results.json ─────────────────────────────────────────────────
    rjson = RESULTS_ROOT / "ablation_results.json"
    if rjson.exists():
        with open(rjson) as f:
            out["ablation_results"] = json.load(f)
    else:
        out["ablation_results"] = None

    # ── ablation_results_t2.json ──────────────────────────────────────────────
    rjson2 = RESULTS_ROOT / "ablation_results_t2.json"
    if rjson2.exists():
        with open(rjson2) as f:
            out["ablation_results_t2"] = json.load(f)
    else:
        out["ablation_results_t2"] = None

    def _read_db(path):
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM experiments ORDER BY id")
        exps = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM training_history ORDER BY experiment_id, epoch")
        hist = [dict(r) for r in cur.fetchall()]
        conn.close()
        return exps, hist

    # ── experiments.db (Trial 1) ──────────────────────────────────────────────
    db = RESULTS_ROOT / "experiments.db"
    if db.exists():
        out["experiments"], out["training_history"] = _read_db(db)
    else:
        out["experiments"] = []; out["training_history"] = []

    # ── experiments_t2.db (Trial 2) ───────────────────────────────────────────
    db2 = RESULTS_ROOT / "experiments_t2.db"
    if db2.exists():
        out["experiments_t2"], out["training_history_t2"] = _read_db(db2)
    else:
        out["experiments_t2"] = []; out["training_history_t2"] = []

    # ── list all files ────────────────────────────────────────────────────────
    files = []
    for root, dirs, fs in os.walk(RESULTS_ROOT):
        for fn in fs:
            p = Path(root) / fn
            files.append({"path": str(p.relative_to(RESULTS_ROOT)), "size_kb": round(p.stat().st_size / 1024, 1)})
    out["files"] = sorted(files, key=lambda x: x["path"])

    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Pulling results from avatar-results-vol…")
    data = list_and_read.remote()

    out_dir = Path("/Users/anasahmed/AI_Main_Repo/AI_Avatar_detection/AI_Avatar_v2/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save Trial 2 ablation results
    if data.get("ablation_results_t2"):
        p = out_dir / "ablation_results_t2.json"
        with open(p, "w") as f:
            json.dump(data["ablation_results_t2"], f, indent=2)
        print(f"  Saved ablation_results_t2.json  ({p.stat().st_size//1024}KB)")

    if data.get("experiments_t2"):
        p = out_dir / "experiments_t2.json"
        with open(p, "w") as f:
            json.dump(data["experiments_t2"], f, indent=2)
        print(f"  Saved experiments_t2.json  ({len(data['experiments_t2'])} rows)")

    if data.get("training_history_t2"):
        p = out_dir / "training_history_t2.json"
        with open(p, "w") as f:
            json.dump(data["training_history_t2"], f, indent=2)
        print(f"  Saved training_history_t2.json  ({len(data['training_history_t2'])} rows)")

    # save Trial 1 ablation results
    if data["ablation_results"]:
        p = out_dir / "ablation_results.json"
        with open(p, "w") as f:
            json.dump(data["ablation_results"], f, indent=2)
        print(f"  Saved ablation_results.json  ({p.stat().st_size//1024}KB)")

    # save experiments table
    if data["experiments"]:
        p = out_dir / "experiments.json"
        with open(p, "w") as f:
            json.dump(data["experiments"], f, indent=2)
        print(f"  Saved experiments.json  ({len(data['experiments'])} rows)")

    # save training history
    if data["training_history"]:
        p = out_dir / "training_history.json"
        with open(p, "w") as f:
            json.dump(data["training_history"], f, indent=2)
        print(f"  Saved training_history.json  ({len(data['training_history'])} rows)")

    # list volume files
    print("\n  Files on avatar-results-vol:")
    for f in data["files"]:
        print(f"    {f['path']:50s}  {f['size_kb']:>8.1f} KB")

    print("\nDone.")
    return data
