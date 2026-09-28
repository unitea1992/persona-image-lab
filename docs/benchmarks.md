# Benchmarks

Baseline measured September 20, 2026 on ASUS GX10, NVIDIA GB10, approximately
121.6 GiB usable unified memory, driver 580.173.02, NVIDIA PyTorch 26.08 container.
Model revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`;
Diffusers revision `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`.

BF16, 1024 x 1024, 40 steps, true CFG 1.0, KV cache enabled. One full warmup,
then batch order 1, 2, 4, 4, 2, 1, using the same prompt and seeds 42 onward.
CUDA synchronized around inference. No quantization, compilation, or step reduction.

| Batch | Trial times (s) | Mean s/image | Images/min | Peak allocated GiB | Peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 52.716, 52.408 | 52.562 | 1.1415 | 36.915 | 38.322 |
| 2 | 101.997, 101.505 | 50.876 | 1.1793 | 42.976 | 47.332 |
| 4 | 201.069, 201.808 | 50.360 | 1.1914 | 55.687 | 61.072 |

All 14 measured images completed. Batch 4 saves about 8.8 seconds compared with
four sequential baseline images, a 4.4% throughput improvement. This establishes
that four fits for this workload, not the maximum batch size. The app keeps
single-image inference as its interactive default.

Times include prompt encoding, denoising, decode and PIL conversion, excluding
model loading and disk writes. Allocated and reserved PyTorch memory differ;
neither is total device/system usage. Warmup was 54.09 seconds and model loading
197.6 seconds. Different prompts, reference edits, larger resolutions, and other
machines require separate measurements. Do not advertise this as a universal
DGX Spark performance guarantee.

## GB10 max-speed screening, 2026-09-27 (pgx-head, this repo worktree)

1024x1024, batch=1, same prompt/seed gate (pixel std >= 5, non-blank), prefix
KV cache on. Baseline is origin/main b3351f6 regional torch.compile,
re-measured here: cold 59.2s (incl. compile), steady 42.1-42.4s
(mean 42.23s, peak 38.9 GiB allocated).

| Config | Steady 1024/40/b1 | Speedup vs baseline | Peak alloc | Quality note | Startup note |
| --- | --- | --- | --- | --- | --- |
| Baseline BF16 + regional compile | 42.23s | 1.00x | 38.9 GiB | reference | load ~210s |
| torchao FP8 weight-only + regional | 45.29s | 0.93x (slower) | 32.3 GiB | valid, memory-only | warmup 156s |
| torchao FP8 dynamic A8W8 + regional | 34.42s | 1.23x | 32.3 GiB | valid, matches BF16 layout | warmup 235s |
| cuDNN decode backend (BF16) | 41.36s | 1.02x | 38.9 GiB | valid | same as baseline |
| FlexAttention processor + regional | 41.44s | 1.02x | 38.9 GiB | valid | same as baseline |
| VAE decode max-autotune-no-cudagraphs | 40.50s | 1.04x | 34.6 GiB | valid | warmup 191s |
| FBCache t=0.05 + regional | setup-failed | — | — | torch.compile graph break | — |
| FBCache t=0.05 eager | failed | — | — | shape mismatch, dropped | — |
| SGLang Diffusion native (BF16, eager) | 44.5-49.6s | 0.85-0.95x | 19.3 GiB process | valid image, slower here | load ~50s |

Step reduction (BF16 regional, same prompt/seed): 40 steps 42.15s, 32 steps
33.59s, 28 steps 29.45s, 24 steps 25.38s, 20 steps 21.39s — near-linear.
FP8-dynamic + 28 steps: 24.37s steady (1.73x vs baseline 40-step);
FP8-dynamic + 24 steps: 21.49s (1.96x). All saved step-sweep images render
complete scenes with legible plaque text; 28 steps keeps full detail while 24
steps stays usable. The 9-case quality matrix (3 prompts x 3 seeds, incl. one
image-conditioned edit) passes the usability gate for both BF16-28 and
FP8-dynamic-28 with no blank/noise outputs.

Initial-screening candidate (not the final default): PERSONA_FP8=dynamic
+ regional compile + 28 UI steps (~24.4s same-shape steady T2I). It collapsed
on new prompt shapes (see clean mixed-prompt reruns below); the final default
is BF16 regional static (PERSONA_FP8=off). PERSONA_DEFAULT_STEPS / UI slider
still allow 40.

## Review correction, 2026-09-27 night (mixed-prompt workload, clean base)

The screen above reused a pre-quantized/pre-compiled pipe for some configs,
so those rows are kept as history, not as the decision basis. Clean re-runs
start every config from bf16+eager (PERSONA_FP8=off, PERSONA_TORCH_COMPILE=0,
fresh TORCHINDUCTOR_CACHE_DIR) with a contamination assert. Same process runs
warmup + same x2 + short/medium/long at 28 steps, 1024x1024 batch=1:

| Config (clean) | warm | same-1 | short | medium | long | mix4 mean/med/max |
| --- | --- | --- | --- | --- | --- | --- |
| A FP8dyn + regional static | 275.0s | 24.80s | 88.20s | 53.86s | 37.26s | 51.03 / 45.56 / 88.20 |
| F FP8dyn + max-autotune-no-cudagraphs | 260.4s | 25.59s | 102.93s | 69.61s | 40.46s | 59.65 / 55.03 / 102.93 |
| C BF16 + regional dynamic=True | 50.9s | 28.97s | 35.92s | 35.99s | 36.12s | 34.25 / 35.95 / 36.12 |
| E BF16 + regional static | 45.2s | 29.04s | 35.65s | 32.01s | 31.71s | 32.10 / 31.86 / 35.65 |
| G BF16 + max-autotune-no-cudagraphs | 94.2s | 28.98s | 59.07s | 36.52s | 36.92s | 40.37 / 36.72 / 59.07 |
| D FP8dyn eager (no compile) | 53.8s | 51.36s | 51.47s | 51.97s | 51.76s | 51.64 / 51.61 / 51.97 |

All rows valid images (pixel-std gate). Peak: FP8 rows 32.3 GiB, BF16 rows
38.9 GiB, eager rows ~30.3 GiB. dynamic=True on FP8dyn (config B) never
finished warmup (>1h, killed): Float8 dispatch hits unimplemented aten.abs in
the compiled region, so B is dropped as broken, not slow. The earlier
aten.abs failure was therefore a real FP8dyn+dynamic incompatibility, not
harness contamination.

Decision: mixed-prompt p95/latency dominates real use. E (BF16 static) wins
mix4 mean 32.10s / max 35.65s and needs no FP8. C (BF16 dynamic=True) is
flatter (max 36.12s, warm 50.9s) but slower on average (-2.1s) with no p95
win, so it is not adopted. FP8dyn static (A) has the best same-shape steady
(24.8s) but collapses on new shapes (short 88.2s). max-autotune-no-cudagraphs
(F/G) never beats default-mode regional on mixed prompts; discarded.
24-step BF16 static (H): same 25.1s, short 35.7s, med 29.7s, long 29.6s,
mix4 30.02/29.63/35.68, all valid with legible plaque text — 6.5% faster
than 28-step E, kept as a runtime option (PERSONA_DEFAULT_STEPS=24), while
the shipped default stays 28 for detail margin.

Final default: BF16 + regional static + 28 steps (PERSONA_FP8=off).
PERSONA_FP8=dynamic remains available for repeated same-shape batches.

## Final step Pareto, 2026-09-28 (BF16 static regional, clean base)

Same process per step count: warmup + same/short/medium/long, 1024x1024
batch=1, valid-gated. 20-step run reuses one warmed inductor cache dir per
config; recompile spikes shrink after the first shape of each process:

| Steps | same | short | medium | long | mix4 mean/med/max |
| --- | --- | --- | --- | --- | --- |
| 28 | 29.15s | 39.94s | 33.57s | 33.49s | 34.04 / 33.53 / 39.94 |
| 24 | 25.20s | 25.04s | 24.94s | 25.22s | 25.10 / 25.12 / 25.22 |
| 20 | 21.26s | 20.90s | 20.96s | 21.08s | 21.05 / 21.03 / 21.26 |

Quality matrix at 20 steps (BF16 static, recompile limit 32): 9 T2I
(cube/chrome/JA-portrait x seeds 42/43/44) + night edit + transparent
extract, 11/11 complete, all valid. No blank/noise/NaN; faces, plaque text
'LOCAL WORLD', chrome-J geometry, night-edit composition, and transparent
isolation all usable; differences vs 24/28 are fine detail only. 20-step T2I
mean 21.1s (median 21.1s), edits 32.8s/32.0s.

Recompile-limit fix: fullgraph=True regional blocks hit Dynamo's default
per-frame recompile limit (8) on long mixed T2I+I2I sessions
(FailOnRecompileLimitHit, step quality unrelated). Root cause per
torch/_dynamo/convert_frame.py: any 9th guard-set recompile of the same
block forward hard-fails under fullgraph. Fix (GB10-only, in
`_configure_transformer_acceleration`): raise `recompile_limit` and
`cache_size_limit` 8 -> 32 when lower; no unbounded setting. Verified: the
same 9-T2I + 2-I2I 20-step session that failed at limit 8 completes at 32
with peak 40.9 GiB and no abnormal compile growth (first-seen shapes pay
once, repeats stay ~21s).

Adopted final default: BF16 + regional static + 20 steps
(PERSONA_FP8=off, PERSONA_DEFAULT_STEPS=20). 24/28 stay selectable at
runtime; FP8-dynamic stays opt-in for repeated same-shape batches.

## Reproduce

Stop the app first to avoid loading two model instances. Run on the Spark:

```bash
./persona stop
docker compose run --rm --no-deps lab python benchmark_batches.py
./persona start
```

Export `LOCAL_UID=$(id -u)` and `LOCAL_GID=$(id -g)` before the direct Compose
command when your UID/GID differ from 1000. Results and images are written under
`outputs/batch-benchmark-<timestamp>/`, excluded from Git. The script uses
`demos.json` only as a reproducible benchmark fixture; demos are not UI presets.
Review a benchmark's metadata for private paths before sharing it.
