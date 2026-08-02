# Exact token-weighted mixture for NVIDIA Nemotron-ClimbMix

This repository publishes a reproducible full-release scan of the official GPT-2-tokenized [`nvidia/Nemotron-ClimbMix`](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) files.

The main output is the exact per-cluster sum of the release's `token_count` metadata at immutable revision:

```text
5eaa64b9c0c85b7f56af01d7dffdb0795816b12b
```

It exists so users who need **token-budgeted** sampling weights do not have to transfer and scan roughly 2 TB again.

## Key results

- 100 root `part_*.tokenized.jsonl` files
- 1,987,970,304,099 source bytes
- 553,315,056 documents
- 356,864,528,972 GPT-2 source tokens
- 351,792,454,745 tokens after excluding cluster 11
- 5,072,074,227 cluster-11 tokens, or 1.421288% of the released source tokens

The accepted-cluster weight file is [`results/climbmix_code_free_weights.json`](results/climbmix_code_free_weights.json), with SHA-256:

```text
76e82e22760adcac59c7294fe9bac11358f5a8b7a26035aae64c3f2e6fa1acb7
```

The filename is historical and imprecise. The actual guarantee is **programming-cluster-excluded**, not code-free.

## Exact cluster totals

`All-token share` is relative to all 20 clusters. `Accepted share` is conditioned on excluding cluster 11.

| Cluster | Documents | Source tokens | All-token share | Accepted share |
|---:|---:|---:|---:|---:|
| 1 | 4,785,103 | 3,063,924,776 | 0.858568% | 0.870947% |
| 2 | 6,684,586 | 4,582,654,670 | 1.284144% | 1.302659% |
| 3 | 8,003,099 | 5,771,522,873 | 1.617287% | 1.640605% |
| 4 | 21,348,980 | 12,797,349,739 | 3.586053% | 3.637756% |
| 5 | 10,450,928 | 6,563,519,102 | 1.839219% | 1.865736% |
| 6 | 98,368,523 | 70,958,523,234 | 19.883882% | 20.170564% |
| 7 | 92,561,323 | 64,041,261,564 | 17.945539% | 18.204274% |
| 8 | 6,455,507 | 3,818,960,760 | 1.070143% | 1.085572% |
| 9 | 4,493,536 | 2,978,940,628 | 0.834754% | 0.846789% |
| 10 | 40,603,579 | 26,975,813,961 | 7.559119% | 7.668105% |
| 11 | 8,630,635 | 5,072,074,227 | 1.421288% | excluded |
| 12 | 142,111,098 | 78,310,785,713 | 21.944121% | 22.260507% |
| 13 | 5,004,064 | 2,561,634,721 | 0.717817% | 0.728166% |
| 14 | 1,530,996 | 834,424,339 | 0.233821% | 0.237192% |
| 15 | 1,296,383 | 804,163,375 | 0.225341% | 0.228590% |
| 16 | 40,297,278 | 27,851,136,242 | 7.804400% | 7.916923% |
| 17 | 38,854,459 | 26,785,891,472 | 7.505899% | 7.614118% |
| 18 | 12,586,375 | 7,392,477,294 | 2.071508% | 2.101375% |
| 19 | 6,437,288 | 3,836,811,747 | 1.075145% | 1.090646% |
| 20 | 2,811,316 | 1,862,658,535 | 0.521951% | 0.529477% |

## Why this is different from existing public work

[`gvlassis/ClimbMix`](https://huggingface.co/datasets/gvlassis/ClimbMix) already provides a very useful detokenized, cluster-separated release and exact **document-count** ratios. Its all-cluster document counts match this scan exactly, including the total of 553,315,056 documents.

Document ratios are not token ratios: document lengths differ by cluster. For token-budgeted pretraining, scheduler weights should normally be based on source-token totals. This repository adds that token-weighted view and publishes the exact integer weights.

## Verification

The result was accepted only after all of the following passed:

1. The pinned Hugging Face tree was re-resolved at the immutable revision; all 100 paths, sizes, and the total byte count exactly matched the published artifacts.
2. Exact work-plan coverage of every byte in all 100 pinned source files, with no gaps or overlaps.
3. Exact-once JSONL ownership at adjacent byte-range boundaries, including records much larger than a region and a final record without a newline.
4. 31 focused tests covering boundary ownership, deterministic work plans, retries, fail-closed metadata parsing, interruption/resume equivalence, and drift rejection.
5. Final report, progress, weights, and work-plan hash agreement.
6. A completed `--resume` check producing byte-identical artifacts.
7. An independent full-JSON sample of 1,000 records—10 from each source file—with zero `token_count != len(tokens)` or token-ID-bound mismatches.
8. Exact equality with the independently published per-cluster document counts from `gvlassis/ClimbMix`.

Full details are in [`results/verification.json`](results/verification.json). Run the offline verifier with:

```bash
python -m pip install -e .
climbmix-verify results
python -m unittest discover -v
```

## Reproduce the full scan

A fresh run transfers the full pinned release and can take several days depending on the network path:

```bash
python -m pip install -e .
climbmix-mixture \
  --output-dir ./calibration-output \
  --workers 8 \
  --max-in-flight-work-items 16
```

Resume after interruption with the same source policy:

```bash
climbmix-mixture \
  --output-dir ./calibration-output \
  --workers 8 \
  --max-in-flight-work-items 16 \
  --resume
```

The scanner reads only `cluster_id` and `token_count` from each pinned JSONL record and does not materialize the large token arrays during the full pass. It fails closed on source-size changes, malformed layout, work-plan drift, missing clusters, or hash mismatch.

## Scope and limitations

- These are exact sums of the official `token_count` metadata, not an exhaustive second count of every token array.
- The metadata-to-array relationship was independently checked on 1,000 fully parsed records across all 100 source files with no mismatch.
- Results apply only to the pinned revision and root tokenized files listed in the published work plan.
- No raw documents or token arrays are redistributed here.
- Cluster 11 is NVIDIA's explicit programming/software cluster, but other broad clusters can still contain incidental code.
- This repository does not claim that the source topic labels are perfectly pure.

## Attribution

Nemotron-ClimbMix is published by NVIDIA for research and development under CC BY-NC 4.0. See [`NOTICE.md`](NOTICE.md) and the official dataset card. The calibration code in this repository is MIT-licensed.

Please cite the original CLIMB paper:

> Shizhe Diao et al. “CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training.” arXiv:2504.13161, 2025.
