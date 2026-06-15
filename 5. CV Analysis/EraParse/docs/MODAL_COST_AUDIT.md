# Modal Cost Audit

## Scope

Actual Modal billing from June 9 through June 11, 2026, queried with:

```bash
modal billing report --start 2026-06-09 --end 2026-06-12 --json
```

## Actual Spend

| App family | Runs | Cost |
|---|---:|---:|
| Qwen3 trials | 11 | $10.77 |
| Donut inference trials | 8 | $8.99 |
| Donut training | 12 | $5.75 |
| NuExtract trials | 8 | $1.16 |
| document representations | 2 | $0.02 |
| **Total** | **41** | **$26.69** |

Most cost came from real full-split model inference and full Donut training.
However, implementation mistakes caused avoidable paid work.

## Directly Attributable Avoidable Spend

Documented failed, stopped, non-resumable, or invalid-target-contract runs
account for at least **$5.20**:

- non-resumable Qwen3 validation timeout;
- Donut gradient-accumulation and checkpoint-policy stops;
- Donut import failure and Modal-cancelled full attempt;
- raw and native Donut full training runs invalidated by the duplicated task
  prompt label contract.

This is a lower bound. Several Donut inference runs used invalidated
checkpoints, but existing App names do not uniquely identify their purposes, so
their costs are not assigned without evidence.

## Corrections

- Full inference must persist each completed chunk.
- Full training requires checkpoint save/reload and tiny-set overfit gates.
- Fine-tuning promotion requires generated-output validity, not falling loss.
- Smoke, debug, selection, and final runs need distinct App names or billing
  tags.
- Every paid run records expected maximum runtime, stop rule, and actual
  post-run billing cost.
- Multiple GPUs are prohibited unless the implementation explicitly uses them.
  Two T4s do not automatically pool memory and currently cost slightly more
  than one A10.

The shared Modal skill now encodes these safeguards.
