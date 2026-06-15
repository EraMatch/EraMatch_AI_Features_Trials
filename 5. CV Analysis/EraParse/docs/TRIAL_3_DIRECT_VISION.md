# Trial 3 Direct Vision Stage

## Status

Completed on June 11, 2026.

This stage added the missing OCR-recovery control and tested the direct-vision
baseline required before SG-VTC architecture edits. The OCR-recovery lane
passed selection. Donut-base did not pass the schema-generation gate, so
SG-VTC remains blocked.

Active/completed Modal runs:

- OCR-fallback Qwen3 `debug_50`: `ap-0SAYtb5zGrfnidYmmCwq7j`
- OCR-fallback Qwen3 full validation: `ap-F6wkVOMDG2RewnSjtkRjG4`
- Donut initial two-step smoke: `ap-Crp81k7ImkcON3r92HXEv1`
- Donut corrected two-step smoke: `ap-cWFNp0FpHDlwLAweoUjtHu`
- Donut epoch-checkpoint smoke: `ap-UeefveO0kGLwxSMKLDERBT`
- Donut full-token inference smoke: `ap-bPU5shLEZOakKR1RsGM9S2`
- Donut raw-JSON full 1,445-sample training: `ap-3G2Xzj2GkUY31s277YsbCs`
- Donut native-token smoke: `ap-Jgljievf91X9P0TfelHWRD`
- Donut corrected native-token smoke: `ap-uBKwNnlX7ceqQLW37tQrjC`
- Donut corrected native-token full training: `ap-JFpLEjTpJbs94KTOyl05U4`
- Donut corrected native-token epoch-3 validation: `ap-9hjsiXxlXJBzOiS8qLwX0v`
- Donut decoding-control spike: `ap-keIYCnroPuD9O3mSBKojYw`

## Stage 3A: OCR Recovery Control

Representation: `pymupdf_tesseract_fallback`

Policy:

1. use precomputed PyMuPDF text when it contains non-whitespace text;
2. otherwise use the precomputed Tesseract OCR artifact;
3. pass the resulting evidence to the existing pinned Qwen3 mapper without
   changing its prompt or generation settings.

This is a controlled representation intervention. It tests whether T4 failure
is primarily missing text rather than insufficient mapper capability.

Promotion sequence:

1. balanced `debug_50`;
2. optional `debug_250` only if failure behavior is unclear;
3. full 310 validation for selection;
4. protected tests only after freezing.

The `debug_50` gate passed:

| Lane | Documents | Macro | JSON/schema valid | Unsupported evidence |
|---|---:|---:|---:|---:|
| Qwen3/PyMuPDF | 50 | 0.5593 | 98.00% | 2.71% |
| Qwen3/PyMuPDF with Tesseract fallback | 50 | 0.6462 | 98.00% | 4.47% |

The fallback removes all ten empty T4 inputs in `debug_50` and improves macro
score by 0.0869. Its higher unsupported-evidence rate remains a selection
criterion. The lane has been promoted to all 310 validation CVs; no protected
test data is being used.

Tracked compact debug summary:
`reports/models/qwen3-pymupdf-tesseract-fallback-debug50-summary.json`

Full validation confirms the promotion:

| Documents | Macro | JSON valid | Schema valid | Unsupported evidence |
|---:|---:|---:|---:|---:|
| 310 | 0.6709 | 100.00% | 99.68% | 4.10% |

Validation macro by tier:

| Tier | Documents | Macro |
|---|---:|---:|
| T1 | 91 | 0.7349 |
| T2 | 93 | 0.7388 |
| T3 | 48 | 0.6097 |
| T4 | 55 | 0.5104 |
| T5 | 23 | 0.6549 |

No validation response reached the 1,600-token cap. The lane is frozen in
`configs/models/qwen3_pymupdf_tesseract_fallback_selected_v1.json` for later
final comparison. It must not run on protected tests until the direct-vision
selection stage is complete.

Tracked compact validation summary:
`reports/models/qwen3-pymupdf-tesseract-fallback-validation-summary.json`

## Stage 3B: Donut Fine-Tuning

Model: `naver-clova-ix/donut-base`

Pinned revision: `a959cf33c20e09215873e338299c900f57047c61`

Frozen initial configuration:
`configs/models/donut_base_finetune_v1.json`

Training contract:

- train only on the 1,445 training CVs;
- select using the 310 validation CVs;
- serialize the reduced CV target as deterministic compact JSON;
- use `<s_eraparse>` as the task prompt;
- upload exact local page-image bytes to the dedicated `eraparse-dataset`
  Modal volume;
- write checkpoints to `eraparse-checkpoints`;
- record target-token truncation before accepting a run.

Multi-page baseline policy: vertically stack pages in manifest reading order
with a white separator, then apply the pinned Donut processor. This is simple
and reproducible but may reduce effective text resolution for two-page CVs.
It must be reported as a limitation and can later be compared with page-wise
encoding or learned aggregation.

## Stop Gates

The Donut smoke run must demonstrate:

- the pinned checkpoint and processor load;
- exact uploaded images are readable;
- forward/backward optimization completes;
- validation loss computes;
- checkpoint and processor save successfully;
- target truncation is measured.

Do not begin SG-VTC edits until a full-token Donut baseline trains and produces
valid validation predictions. SG-VTC must change only the encoder-output/token
path while preserving the trained target, processor, evaluator, and split.

The corrected two-step smoke run passed all infrastructure gates:

| Measure | Result |
|---|---:|
| training records | 50 |
| validation-loss records | 50 |
| optimizer steps | 2 |
| maximum target tokens | 820 |
| truncated targets | 0 |
| mean training loss | 7.8480 |
| mean validation loss | 6.6208 |

Checkpoint: `/checkpoints/donut-base-smoke-debug50-v3` on the
`eraparse-checkpoints` Modal volume.

The promoted full run uses all 1,445 training records, all 310 validation
records for final loss measurement, three epochs, gradient accumulation of
eight, and no protected test data.

The inference smoke also passed. The unchanged full-token encoder produced
4,800 visual tokens for the sampled CV. Measured latency was 1.6804 seconds for
the encoder and 0.2826 seconds for the decoder at a deliberately short
64-token smoke limit. These are infrastructure measurements, not model-quality
results.

## Excluded Donut Target-Contract Runs

The first raw-JSON and native-token Donut adaptations incorrectly included
`<s_eraparse>` in the label sequence while also configuring it as
`decoder_start_token_id`. The reference Donut fine-tuning contract uses the
task token only to start decoding; labels contain the structured target plus
EOS.

These runs are excluded implementation trials, not model-quality baselines:

- raw JSON epoch 2: validation loss 0.2427, but 310/310 generations hit the cap
  and JSON validity was 0%;
- native-token epoch 3: validation loss 0.4003, but 306/310 generations hit the
  cap and schema validity was 0%.

They still demonstrate an important evaluation lesson: token-level validation
loss is insufficient for structured generation selection.

Reference implementation:
[Fine-tune Donut on CORD](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_%28CORD%29_with_PyTorch_Lightning.ipynb)

## Corrected Native Structural-Token Edit

The architecture-corrected lane uses Donut-native structural target tokens:

- `<s_field>` and `</s_field>` delimit schema fields;
- `<sep/>` separates array items;
- `<s_eraparse>` is used only as the decoder start token and is excluded from
  labels;
- `processor.token2json()` reconstructs the prediction;
- the same image processor, model revision, splits, evaluator, and full-token
  encoder remain unchanged.

This follows the official Donut parsing interface documented in the
[Transformers Donut guide](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/donut).

The corrected native-token smoke passed with 54 added structural tokens,
maximum target length 438, and zero truncation. Full corrected native-token
fine-tuning used all 1,445 training CVs and all 310 validation CVs:

| Epoch | Mean training loss | Mean validation loss |
|---:|---:|---:|
| 1 | 4.7140 | 1.5707 |
| 2 | 1.1460 | 0.6130 |
| 3 | 0.5389 | 0.3882 |

The run completed 543 optimizer steps. Its longest target was 499 tokens and
no targets were truncated.

Epoch 3 failed the structured-generation gate:

| Documents | Macro | JSON valid | Schema valid | Hit 1,536-token cap |
|---:|---:|---:|---:|---:|
| 310 | 0.1295 | 100.00% | 0.00% | 306 |

Deterministic `token2json()` made the generations JSON-parseable, but the
decoded structures did not satisfy the reduced CV schema. Mean visual-token
count was 4,800. Mean encoder, decoder, and total latency were 0.2371, 6.7400,
and 6.9789 seconds respectively.

A 50-document decoding-control spike used a 768-token maximum,
`no_repeat_ngram_size=3`, and `repetition_penalty=1.2`. It removed token-cap
hits but did not restore schema-valid structure, so it was not promoted.

Tracked summaries:

- `reports/models/donut-native-v3-full-train-v1-summary.json`
- `reports/models/donut-native-v3-epoch03-validation-summary.json`
- `reports/models/donut-base-full-train-v4-summary.json`
- `reports/models/donut-raw-json-epoch02-validation-summary.json`
- `reports/models/donut-native-full-train-v1-summary.json`
- `reports/models/donut-native-epoch03-validation-summary.json`

## Stage Decision

The selected practical Trial 3 lane is Qwen3 with
`pymupdf_tesseract_fallback`, with validation macro 0.6709. Donut-base remains
an informative full-token visual baseline implementation, but not an accepted
quality baseline because it failed schema validity.

The next model stage must establish a stronger visual-generation reference
using a modern document VLM upper bound or a constrained/decomposed visual
decoder. SG-VTC architecture edits remain blocked until a full-token visual
lane generates schema-valid predictions. No ID, template-OOD, or locked
confirmation samples were used in this stage.

## Relationship To The Grounded SG-ESE Findings

The later SG-ESE stage found evidence-alignment, source-label, record-grouping,
punctuation, and skill-decoding issues. These did not cause the Donut failures:
Donut trained from images and serialized reduced-schema targets, not from the
new evidence graphs or token labels.

The SG-ESE findings do, however, explain why the corrected Donut contract was
hard. Repeated-record grouping, set-valued skill segmentation, rare-field
imbalance, perception, and schema assembly are distinct problems. Corrected
Donut had to solve all of them jointly through autoregressive generation.
SG-ESE separates them and gives deterministic assembly ownership of the final
schema.

See `SG_ESE_STAGE_LOG.md` for the full causality analysis and current Donut
follow-up requirements.

## Failed And Stopped Runs

Failures are retained as part of the reproducibility record:

- `ap-d3TsQQmU93Tot0VoGctOR7`: the first full run was manually stopped after
  54 optimizer steps because the training loop would carry an incomplete
  gradient-accumulation remainder across epochs. No checkpoint from this run is
  accepted.
- `ap-Btpsc3JUyzgazwghntc89u`: corrected-loop smoke failed before training
  because a dependency-free helper was initially imported through a module
  that also required Pydantic. The helper was isolated and the smoke rerun.
- `ap-uCbieUSKyRyXiJW9IuAkes`: the corrected accumulation run was manually
  stopped because it saved only the final epoch, preventing auditable
  validation-based epoch selection.
- `ap-pyNBsE8kLSZ8MIv5AF7hoY`: the epoch-checkpointed full run was cancelled by
  Modal during epoch 1 before an accepted checkpoint was written.

## Reproducibility Correction

An older Modal volume named `cv-parsing-data` contains different bytes for at
least one matching page-image filename and is not used by this stage. Trial 3
uses a dedicated EraParse volume populated from the audited local dataset.
