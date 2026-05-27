# Ring Joint SDPA Chunked-Prefill Non-Determinism

Investigation report. Branch `skrstic/ring_joint_sdpa_chunked_determinism`, HEAD `32e7a9e32c0`.
Date: 2026-05-27.

## Summary

`ttnn.transformer.ring_joint_scaled_dot_product_attention` produces non-deterministic
output in **chunked-prefill mode** under specific shape combinations: same inputs
uploaded fresh each call, same op invocation, but per-chunk output differs across
otherwise-identical iterations of the chunked loop.

True race — diff count varies run-to-run and the magnitude (up to 0.25 in bf16) is
far above 1 ULP, suggesting a stale/uninitialised read rather than accumulation-order
rounding noise.

The non-chunked per-call ring SDPA determinism test passes on the same configs.
**Confirmed via control experiment**: forcing `use_streaming_compute = false` in
`ring_joint_sdpa_program_factory.cpp:383` restores determinism (3/3 passes). The
race is in the streaming compute path added in PR #44929, not in single-call SDPA.

## Trigger (necessary conditions)

1. Chunked-prefill mode active (`tensor_args.is_chunked() == true`).
2. **Odd `Sq_chunk_t`** (q_chunk is a multiple of 32 but not 64). The host subblock
   solver picks `qk_out_subblock_h = 1` for odd Sq, giving:
   - `qk_in0_num_subblocks = Sq_chunk_t` (3 or 5, both > 2 — Phase 1's per-row-group
     loop runs ≥ 3 times).
   - `has_qktv_remainder = true`. `streaming_qktv_h` returns 2 when the host picked
     `out_out_subblock_h = 1`, so Phase 2 takes the **hoisted-drain remainder path**.
   Both axes flip simultaneously at odd Sq — see "Where the race lives".
3. `k_chunk_size = 256` (`Sk_chunk_t = 8`). In our chunk layout this is the only
   `k_chunk` that produces a partial trailing K-chunk.
4. **Partial trailing Sk-chunk** on the local K shard:
   `kv_local_padded_Nt % Sk_chunk_t != 0`.

Removing any one of (2), (3), (4) makes the op deterministic in our tests.

## Symptoms

- `torch.equal(iter0_chunk_i, iter1_chunk_i)` returns false for the affected chunk
  only; earlier and later chunks remain bit-exact.
- Diff count varies across runs (q=96/k=256, chunk 6: ~38k–47k diffs over repeats).
- Max abs diff up to **0.25** (q=96), as small as **0.015625** (q=160) — both far
  above 1 bf16 ULP (≈9.77e-4).
- Repeated runs always pinpoint the same chunk index for a given (q, k).

## Test matrix (local, sp_size=4, kimi MLA: nhq=16, nhk=1, d_q=576, d_v=128, bf16 Q / bf8 K/V)

Chunked-prefill setup: `chunk_size = 2560` global (per-device per-chunk Q = 640),
7 chunks (chunks 0..6), 2 iterations.

| q_chunk | Sq_chunk_t | k_chunk | Sk_chunk_t | streaming use\* | result   | failing chunks | max diff |
|--------:|-----------:|--------:|-----------:|:---------------:|:--------:|:--------------:|:--------:|
|  64     | 2 (even)   | 256     |  8         | OFF (num_sb=1)  | **PASS** | —              | —        |
|  96     | 3 (odd)    | 256     |  8         | ON  (num_sb=3)  | **FAIL** | 6              | 0.25     |
| 128     | 4 (even)   | 256     |  8         | ON  (num_sb=2)  | **PASS** | —              | —        |
| 160     | 5 (odd)    | 256     |  8         | ON  (num_sb=5)  | **FAIL** | 0 and 6        | 0.047    |
|  96     | 3 (odd)    | 128     |  4         | n/a (k clean)   | **PASS** | —              | —        |
|  96     | 3 (odd)    | 320     | 10         | n/a (k clean)   | **PASS** | —              | —        |
|  96     | 3 (odd)    | 256     |  8         | **streaming OFF** patch | **PASS** | — | —     |

\* "num_sb" = `qk_in0_num_subblocks`. Last row = control experiment (forced
`use_streaming_compute = false`).

## Cross-architecture data (CI run 26507052805, current branch)

Different model (mla_100k: nhq=29, q_chunk=160 → Sq=5 odd), same `k_chunk=256`:

| Topology    | sp_size | model     | q_chunk | k_chunk | failing chunk | # diffs (sampled) | max diff             |
|-------------|--------:|-----------|--------:|--------:|--------------:|------------------:|---------------------:|
| bh_deskbox  | 2       | mla_100k  | 160     | 256     | 1             |    511            | 9.77e-4 (1 bf16 ULP) |
| bh_p300     | 2       | mla_100k  | 160     | 256     | 1             |    383            | 9.77e-4              |
| bh_llmbox   | 4       | mla_100k  | 160     | 256     | none          |      —            | —                    |
| local       | 4       | kimi_100k |  96     | 256     | 6             | ~4 × 10⁴          | 0.25                 |

Per-device K shapes that hit a partial trailing Sk-chunk differ across topologies
and chunk-layouts, which explains why (q=160, k=256) is not exposed on bh_llmbox
but is exposed locally with the kimi layout.

## Smallest reproducer

**Hardware:** BH 4-device ring (sp_size=4) — e.g. local BH quietbox.

**Test file already configured on this branch** (`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`):

| Setting                                                                       | Value          |
|-------------------------------------------------------------------------------|----------------|
| `CHUNKED_PREFILL_PER_DEVICE_CHUNK`                                            | `640`          |
| `CHUNKED_PREFILL_N_CHUNKS`                                                    | `7`            |
| `CHUNKED_PREFILL_MODEL_CONFIGS["kimi_100k"].q_chunk_sizes`                    | `[96]`         |
| `CHUNKED_PREFILL_MODEL_CONFIGS["kimi_100k"].k_chunk_sizes`                    | `[256]`        |
| `test_ring_joint_attention_sdpa_chunked_determinism` → `num_iterations`       | `2`            |

**Repro command** (run from repo root):

```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_chunked_determinism
```

**Race signature** (run the command 3× — the bug is a true race, not just a wrong
result; the diff count drifts while the failing chunk stays at 6):

| Run | failing iter | failing chunk | # diffs   | max diff   |
|-----|--------------|--------------:|----------:|-----------:|
| 1   | 1            | 6             | 37,245    | 0.125      |
| 2   | 1            | 6             | 42,617    | 0.09375    |
| 3   | 1            | 6             | 47,737    | 0.1875     |

(Numbers vary per host / per run; what's reproducible is the failing
`iter=1, chunk=6` and the run-to-run drift in diff count. Chunks 0..5 are
bit-exact across the two iterations.)

**Control experiment** — force-disable streaming compute and re-run:

```cpp
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:383
const bool use_streaming_compute = false;  // was the original 4-clause expression
```

Then `./build_metal.sh` and re-run the repro command. Expected: 3/3 PASS,
confirming the race lives in the streaming compute path.

## Why k_chunk = 256 specifically (in our layout)

Per-device pre-AllGather K rows after chunk `i` = `(i+1) * (chunk_size / sp_size)`
= `(i+1) * 640` in our config. The `kv_local_padded_Nt % Sk_chunk_t != 0`
("partial trailing Sk-chunk") condition becomes:

- `k_chunk = 128`, `Sk = 4`:  `640 / 128 = 5` → never partial.
- `k_chunk = 256`, `Sk = 8`:  `640 / 256 = 2.5` → partial at every even chunk
  index (0, 2, 4, 6, 8, 10).
- `k_chunk = 320`, `Sk = 10`: `640 / 320 = 2` → never partial.

So in our layout only `k_chunk = 256` exercises the partial-K code path at all.

## Where the race lives

The buggy configs (Sq=3, Sq=5) share two structural properties simultaneously, both
flipped by odd Sq via the host subblock solver:

1. **`qk_in0_num_subblocks > 2`** — Phase 1 sub_exp/matmul interleaves over 3 or 5
   row groups instead of 2.
2. **`has_qktv_remainder = true`** — `streaming_qktv_h(1, w, dst, Sq)` returns 2 when
   the host picks `out_out_subblock_h = 1`, but `qktv_q_num_subblocks = Sq / 2` is
   not exact ⇒ `qktv_remainder_h = 1`. Phase 2 takes the **hoisted-drain branch**
   in `compute_streaming.hpp:1146-1174` (otherwise dead code for even Sq ≥ 4).

These two axes cannot be separated from the existing test data (Sq=4 has num_sb=2
AND no remainder; Sq=3/5 have num_sb>2 AND remainder=1). The remainder path is the
prime suspect because (a) it's only exercised in the failing configs and (b) it
contains the most intricate ordering: two SALADs + two pushes + a `sub_exp_first_col_blocks`
all interleaved on `cb_exp_max_diff`, `prev.out`, `prev.sum`, and `out_cb`.

Mostly-covered hazards I traced and ruled out as the *direct* cause:

- **Phase 1 → Phase 2 cb_qkt_im visibility.** `cb_push_back_hold_wr_ptr` advances
  the PACK→UNPACK tile counter; Phase 2's cb_wait_front is satisfied. ✓
- **Phase 2 q_sub=0 drain → q_sub=0 V matmul.** Sub_exp writes row `Sq-1`; V
  matmul reads rows `0..qktv_h-1`. Disjoint positions. ✓
- **Phase 2 q_sub=0 drain → q_sub=1 V matmul (remainder).** Both touch row `Sq-1`
  cols `0..active_Sk-1`. Explicit PACK→UNPACK semaphore handshake at
  `compute_streaming.hpp:1036-1038`. ✓
- **Phase 1 mask l1-acc onto stale `cb_qkt_im[*, active_Sk..Sk_chunk_t)`.** Stale
  values are post-sub_exp from the previous K-chunk; `stale + (-inf) = -inf` is
  deterministic *if* `llk_pack_reconfig_l1_acc(1)` is a clean RMW. Worth direct
  verification.

Open suspects:

- **Indexing arithmetic in the hoisted-drain block** (`compute_streaming.hpp:1146-1174`):
  `salad_correct_fused(..., sum_q_subblock=drain_salad_row=2, ...)` runs after the
  first SALAD has popped `qktv_h` of `cb_exp_max_diff`. The cumulative `cb_wait_front`
  inside `salad_correct_fused` uses `(sum_q_subblock + 1) * sbh`. If the
  remainder/partial-K interaction perturbs the expected fronted count by even one
  tile, reads land on stale data.
- **`cb_exp_max_diff` write at `drain_salad_row` while previous SALAD is still in
  flight.** `sub_exp_first_col_blocks` for the drain uses `pack_tile<false>` at the
  current `wr_ptr` (sequential), but the salad that pops 2 tiles before it doesn't
  insert an explicit PACK barrier.

## Suggested next experiments (in order)

1. **Isolate remainder vs. num_subblocks.** Add a host-side `has_qktv_remainder`
   flag (re-derive via `streaming_qktv_h`) and gate `use_streaming_compute = false`
   on `has_qktv_remainder` alone. Re-run q=96/k=256:
   - PASS → bug is in the remainder/hoisted-drain path. Audit
     `compute_streaming.hpp:1146-1174` and `salad_correct_fused`.
   - FAIL → bug is in the larger num_sb (>2) path. Audit Phase 1's
     interleaved sub_exp/matmul/mask sequence.
2. **Stale-data hypothesis.** Memset (or l1-acc-zero) `cb_qkt_im[*, active_Sk..Sk_chunk_t)`
   to a known value before the mask l1-acc on the partial K-chunk. If determinism
   returns, root cause is mask-onto-stale.
3. **Extra PACK→UNPACK barrier after Phase 1 mask stamp.** Insert a semaphore
   handshake between `apply_lightweight_mask_streaming` and `reduce_c_row_group`
   for the partial K-chunk's last q_subblock. If it fixes it, the mask l1-acc →
   reduce path is racy.

## Note for triage

This is a latent bug exposed by the new chunked-prefill determinism test added in
commit `76424224e` on this branch. The bug exists on `main` (PR #44929,
`5b9b5b73bf7`) — it's not introduced by this branch. The test makes it observable.
