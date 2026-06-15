# Model Catalog

Checked against live Hugging Face and Ollama sources on 2026-06-09. The Qwen3
revision was rechecked on 2026-06-11. Before reportable runs, pin every model
to a tested `revision` and record it.

## Registry

| Role | Exact ID | Approx. size | Architecture / runtime | License | Status |
|---|---|---:|---|---|---|
| extraction-specialized mapper | `numind/NuExtract-1.5-tiny` | 494M | Qwen2/Transformers | MIT | core |
| small general mapper | `Qwen/Qwen3-0.6B` | HF reports 751.6M | Qwen3/Transformers | Apache-2.0 | core |
| stronger conditional mapper | `microsoft/Phi-4-mini-instruct` | 3.8B | Phi3-compatible/Transformers | MIT | conditional |
| direct VLM baseline | `naver-clova-ix/donut-base` | base model | vision-encoder-decoder | MIT | core, fine-tune required |
| token classifier | `microsoft/layoutlmv3-base` | 125.3M | LayoutLMv3/Transformers | CC-BY-NC-SA-4.0 | research-only core |
| modern structured VLM | `numind/NuExtract3` | 4.54B | Qwen3.5 VLM/Transformers | Apache-2.0 | upper bound |
| modern document parser VLM | `PaddlePaddle/PaddleOCR-VL-1.6` | 958.6M | PaddleOCR-VL | Apache-2.0 | upper bound |
| cloud repair/upper bound | `gemma3:12b-cloud` | 12B | Ollama Cloud | Gemma terms | selected subsets |
| document parser | `ibm-granite/granite-docling-258M` | 257.5M | Idefics3/Transformers | Apache-2.0 | optional |
| document VLM | `PaddlePaddle/PP-DocBee-2B` | 2B family | Paddle/Qwen2-VL | Apache-2.0 | optional |

## Reproducibility Rule

Implementation config must include:

```yaml
model_id: "Qwen/Qwen3-0.6B"
revision: "<tested-hugging-face-commit>"
environment: "core-transformers4"
dtype: "bfloat16"
quantization: null
generation:
  do_sample: false
  max_new_tokens: 2048
```

Do not use `trust_remote_code=True` without pinning and reviewing the revision.

## NuExtract 1.5 Tiny

Use the model's extraction-specific prompt format and near-zero temperature.
The current repository ID is `numind/NuExtract-1.5-tiny`, even though historical
model-card snippets may show the older alias `numind/NuExtract-tiny-v1.5`.

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "numind/NuExtract-1.5-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=REVISION,
    torch_dtype="auto",
    device_map="auto",
).eval()

template_text = json.dumps(reduced_schema_template, indent=2)
prompt = (
    f"<|input|>\n### Template:\n{template_text}\n"
    f"### Text:\n{document_text}\n\n<|output|>"
)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
generated = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
raw = tokenizer.decode(generated[0], skip_special_tokens=True)
prediction_text = raw.split("<|output|>", 1)[1]
```

Validate and cache `prediction_text`; never assume generation is valid JSON.

## Qwen3 0.6B

Use non-thinking mode for structured extraction. The hard switch belongs on
`apply_chat_template`. The tested revision is
`c1899de289a04d12100db370d81485cdf75e47ca`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=REVISION,
    torch_dtype="auto",
    device_map="auto",
).eval()

messages = [{"role": "user", "content": extraction_prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
response_ids = generated[:, inputs.input_ids.shape[1]:]
response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
```

## Phi-4 Mini

Treat as conditional until the quantized environment passes a memory and
quality smoke test. Load with Transformers using the exact ID and pinned
revision. Record quantization config and hardware; do not silently compare a
quantized Phi run with an unquantized baseline.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=REVISION, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
).eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": extraction_prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
```

## Donut

`naver-clova-ix/donut-base` is not a ready CV extractor. Add task/schema tokens
and fine-tune it on the reduced target.

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

model_id = "naver-clova-ix/donut-base"
processor = DonutProcessor.from_pretrained(model_id, revision=REVISION)
model = VisionEncoderDecoderModel.from_pretrained(
    model_id, revision=REVISION
)

pixel_values = processor(image, return_tensors="pt").pixel_values
decoder_input_ids = processor.tokenizer(
    "<s_cv_parse>", add_special_tokens=False, return_tensors="pt"
).input_ids
```

Infer encoder output length and processor resize/padding transforms at runtime.
Do not hardcode a 30x80 grid or 2,400 visual tokens.

## LayoutLMv3

Use token classification, not generation. Provide images, words, normalized
0-1000 boxes, and integer labels. Existing dataset boxes form the source-oracle
lane.

```python
from transformers import AutoModelForTokenClassification, AutoProcessor

model_id = "microsoft/layoutlmv3-base"
processor = AutoProcessor.from_pretrained(
    model_id,
    revision=REVISION,
    apply_ocr=False,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    revision=REVISION,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
encoding = processor(
    image,
    words,
    boxes=normalized_boxes,
    word_labels=integer_labels,
    truncation=True,
    return_overflowing_tokens=True,
    return_tensors="pt",
)
```

Chunk documents over 512 tokens and deterministically assemble predicted spans
into the schema.

## NuExtract3

Use as a modern structured visual upper bound. Its processor accepts extraction
template options directly.

```python
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "numind/NuExtract3"
processor = AutoProcessor.from_pretrained(
    model_id, revision=REVISION, trust_remote_code=True
)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    revision=REVISION,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    template=json.dumps(reduced_schema_template, indent=2),
    enable_thinking=False,
).to(model.device)
```

## PaddleOCR-VL 1.6

Use an isolated PaddleOCR environment. The official page-level parser is the
preferred comparison:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6")
results = pipeline.predict("path/to/document_image.png")
for result in results:
    result.save_to_json(save_path="output")
    result.save_to_markdown(save_path="output")
```

The model card currently requires PaddlePaddle 3.2.1+ and
`paddleocr[doc-parser]>=3.6.0`. Its Transformers path requires Transformers 5.x
and is element-level; keep it in the modern VLM lane.

## Gemma 3 Cloud

Exact command/tag:

```bash
ollama run gemma3:12b-cloud
```

Ollama Cloud currently does not support schema-enforced structured outputs.
Prompt for JSON, validate it locally, cache request/response metadata, and
record repairs and retries. Use only on selected subsets or repair routes.

## Optional Comparator Syntax

Granite-Docling can be invoked through an image-text pipeline:

```python
from transformers import pipeline

parser = pipeline(
    "image-text-to-text",
    model="ibm-granite/granite-docling-258M",
    revision=REVISION,
)
```

PP-DocBee uses PaddleOCR's `DocVLM`, not the core Transformers lane:

```python
from paddleocr import DocVLM

model = DocVLM(model_name="PP-DocBee-2B")
results = model.predict(
    input={"image": image_path, "query": "Parse this CV into Markdown."},
    batch_size=1,
)
```
