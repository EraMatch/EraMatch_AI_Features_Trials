"""
Push a Kaggle kernel via the REST API so we can specify machineShape (e.g. GPU_T4_X2).
The kaggle CLI does not expose machineShape in kernel-metadata.json.

Usage:
    python push.py <script_file> <kernel_slug> <title>

Example:
    python push.py trial_srm_convnext.py avatar-detection-srm-convnext "Avatar Detection - SRM ConvNeXt"
"""
import json, sys, time
import requests
from pathlib import Path


DATASETS = [
    "kaustubhdhote/human-faces-dataset",
    "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",
    "shreyanshpatel1/130k-real-vs-fake-face",
]
MACHINE_SHAPE = "GPU_T4_X2"   # 2× T4 16GB each
API_BASE      = "https://www.kaggle.com/api/v1"

# Numeric kernel IDs (from kernel-metadata.json id_no after first pull)
KERNEL_IDS = {
    "avatar-detection-srm-convnext-ablation": 123056392,
}


def load_creds() -> tuple[str, str]:
    p = Path("~/.kaggle/kaggle.json").expanduser()
    d = json.loads(p.read_text())
    return d["username"], d["key"]


def push(script_path: str, slug: str, title: str,
         machine_shape: str = MACHINE_SHAPE) -> dict:
    username, key = load_creds()
    code = Path(script_path).read_text()

    payload = {
        "text":               code,
        "language":           "python",
        "kernelType":         "script",
        "newTitle":           title,
        "isPrivate":          True,
        "enableGpu":          True,
        "enableInternet":     True,
        "machineShape":       machine_shape,
        "datasetDataSources": DATASETS,
    }
    # Include numeric id when updating an existing kernel (avoids title-conflict 409)
    if slug in KERNEL_IDS:
        payload["id"] = KERNEL_IDS[slug]

    r = requests.post(f"{API_BASE}/kernels/push",
                      auth=(username, key), json=payload)

    if r.status_code != 200:
        print(f"Push failed ({r.status_code}):\n{r.text}")
        sys.exit(1)

    data = r.json()
    ref  = data.get("ref", f"{username}/{slug}")
    print(f"Pushed → https://www.kaggle.com/code/{ref}")
    print(f"machineShape requested: {machine_shape}")
    return data


def poll(slug: str, interval: int = 20, timeout: int = 43200) -> str:
    username, key = load_creds()
    url     = f"{API_BASE}/kernels/{username}/{slug}"
    elapsed = 0

    print(f"\nPolling status every {interval}s (timeout {timeout//3600}h)…")
    while elapsed < timeout:
        r = requests.get(url, auth=(username, key))
        if r.status_code != 200:
            print(f"Status check failed: {r.status_code}")
            time.sleep(interval)
            elapsed += interval
            continue

        status = r.json().get("currentRunningVersion", {}).get("status", "unknown")
        print(f"  [{elapsed//60:3d}m] {status}")

        if status in ("complete", "cancelAck", "error"):
            return status

        time.sleep(interval)
        elapsed += interval

    return "timeout"


def download_output(slug: str, dest: str = "/tmp/kaggle_output") -> None:
    import subprocess
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output",
         f"anasahmad25/{slug}", "-p", dest],
        check=True
    )
    print(f"Output saved to {dest}/")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python push.py <script_file> <slug> <title>")
        sys.exit(1)

    script_file, slug, title = sys.argv[1], sys.argv[2], sys.argv[3:]
    title = " ".join(title)

    push(script_file, slug, title)
    final = poll(slug)
    print(f"\nFinal status: {final}")

    if final == "complete":
        download_output(slug)
