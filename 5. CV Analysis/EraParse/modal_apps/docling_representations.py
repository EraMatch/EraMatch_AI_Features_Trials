import json
import time
from io import BytesIO
from pathlib import Path

import modal

app = modal.App("eraparse-document-representations")
model_volume = modal.Volume.from_name("eraparse-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "docling==2.82.0",
        "pymupdf4llm==1.27.2.2",
    )
)


@app.cls(
    image=image,
    cpu=4,
    memory=16_384,
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class DoclingWorker:
    @modal.enter()
    def load(self) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    @modal.method()
    def convert(self, documents: list[dict]) -> list[dict]:
        import pymupdf
        import pymupdf4llm
        from docling.datamodel.base_models import DocumentStream

        results: list[dict] = []
        for index, item in enumerate(documents, start=1):
            started = time.perf_counter()
            try:
                pdf_bytes = item["pdf_bytes"]
                stream = DocumentStream(
                    name=f"{item['cv_id']}.pdf",
                    stream=BytesIO(pdf_bytes),
                )
                result = self.converter.convert(stream).document
                pymupdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                results.append(
                    {
                        "cv_id": item["cv_id"],
                        "docling_markdown": result.export_to_markdown(),
                        "docling_json": result.export_to_dict(),
                        "pymupdf4llm_markdown": pymupdf4llm.to_markdown(pymupdf_document),
                        "pymupdf4llm_json": pymupdf4llm.to_json(pymupdf_document),
                        "latency_seconds": time.perf_counter() - started,
                    }
                )
            except Exception as error:
                results.append({"cv_id": item["cv_id"], "error": str(error)})
            print(f"converted {index}/{len(documents)}")
        model_volume.commit()
        return results


@app.local_entrypoint()
def main(manifest_path: str, dataset_root: str, output_root: str) -> None:
    manifest = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = Path(dataset_root)
    documents = [
        {
            "cv_id": row["cv_id"],
            "pdf_bytes": (root / row["artifacts"]["pdf"]["path"]).read_bytes(),
        }
        for row in manifest
    ]
    results = DoclingWorker().convert.remote(documents)

    destination = Path(output_root)
    directories = {
        "docling_markdown": destination / "docling_markdown",
        "docling_json": destination / "docling_json",
        "pymupdf4llm_markdown": destination / "pymupdf4llm_markdown",
        "pymupdf4llm_json": destination / "pymupdf4llm_json",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    failures = []
    for result in results:
        if "error" in result:
            failures.append({"cv_id": result["cv_id"], "error": result["error"]})
            continue
        for representation, directory in directories.items():
            suffix = ".json" if representation.endswith("_json") else ".md"
            value = result[representation]
            if suffix == ".json" and isinstance(value, str):
                value = json.loads(value)
            content = json.dumps(value, indent=2) if suffix == ".json" else value
            (directory / f"{result['cv_id']}{suffix}").write_text(content, encoding="utf-8")

    summary = {
        "manifest": manifest_path,
        "completed": len(results) - len(failures),
        "failures": failures,
        "records": [
            {
                "cv_id": result["cv_id"],
                "latency_seconds": result.get("latency_seconds"),
            }
            for result in results
            if "error" not in result
        ],
    }
    for directory in directories.values():
        (directory / "generation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
