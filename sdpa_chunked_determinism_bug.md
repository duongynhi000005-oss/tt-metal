# Ring Joint SDPA Chunked-Prefill Non-Determinism

Investigation report. Branch `skrstic/ring_joint_sdpa_chunked_determinism`.
Date: 2026-05-27.

## Summary

`ttnn.transformer.ring_joint_scaled_dot_product_attention` produces
non-deterministic output in **chunked-prefill mode** under specific shape
combinations: same inputs uploaded fresh each call, same op invocation,
but per-chunk output differs across otherwise-identical iterations of the
chunked loop.

True race — diff count varies run-to-run and the magnitude (up to 0.25 in
bf16) is far above 1 ULP, suggesting a stale/uninitialised read rather
than accumulation-order rounding noise. The race is in the **streaming
compute path** added in PR #44929. Single-call (non-chunked) ring SDPA
determinism tests pass on the same configs.

## Trigger (necessary conditions)

1. Chunked-prefill mode active.
2. **Odd `Sq_chunk_t`** (Sq=3 or Sq=5). Host solver picks
   `qk_out_subblock_h = 1` and `out_out_subblock_h = 1`, so the kernel's
   `qktv_h = 2` and `qktv_remainder_h = Sq % 2 = 1` ⇒
   **`has_qktv_remainder = true`**, Phase 2 takes the hoisted-drain branch
   (`compute_streaming.hpp:1146-1174`).
3. `k_chunk_size = 256` (`Sk_chunk_t = 8`). In the test layout this is the
   only `k_chunk` that produces a partial trailing K-chunk:
   - `k_chunk = 128`, `Sk = 4`:  640 / 128 = 5 → never partial.
   - `k_chunk = 256`, `Sk = 8`:  640 / 256 = 2.5 → partial at even chunk
     indices (0, 2, 4, 6, …).
   - `k_chunk = 320`, `Sk = 10`: 640 / 320 = 2 → never partial.
4. **Partial trailing Sk-chunk** on the local K shard
   (`kv_local_padded_Nt % Sk_chunk_t != 0`).

Removing any one of (2), (3), (4) makes the op deterministic.

## Symptoms

- `torch.equal(iter0_chunk_i, iter1_chunk_i)` returns false for the
  affected chunk only; earlier and later chunks remain bit-exact.
- Diff count varies across runs (q=96/k=256, chunk 6: ~36k–48k diffs).
- Max abs diff up to **0.25** (q=96), as small as **0.0156** (q=160) —
  both far above 1 bf16 ULP (~9.77e-4).
- Repeated runs always pinpoint the same chunk index for a given (q, k).

## Minimal reproducer

**Hardware:** BH 4-device ring (sp_size=4) — e.g. local BH quietbox.

**Test file already configured** on this branch
(`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`):

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

**Race signature** — one run is enough to trigger the failure. The race
is established (diff count drifts across repeats, max stays in the
0.125–0.1875 band, failing chunk stays at 6); during investigation runs
do exactly one repro per hypothesis to read iter / chunk / max-diff
and move on. Representative single-run output:

```
iter 1 chunk 6: ~43,000 diffs, max=0.125
```

Chunks 0..5 are bit-exact across iterations.

## Localization — what's confirmed

The race lives in the **Phase 2 hoisted-drain branch** activated by
`has_qktv_remainder`. Confirmed via three independent experiments — each
toggled OFF the suspected code path and recovered determinism (3/3 PASS
on first build), each ON returned 5/5 FAIL.

### Experiment 1 — host-side disable for remainder configs

Forced `use_streaming_compute = false` only when the kernel would have
taken the remainder branch:

```cpp
// ring_joint_sdpa_program_factory.cpp, near line 383
auto [exp_tentative_out_h, exp_tentative_out_w] =
    detail::determine_largest_subblock_size(Sq_chunk_t, vDHt, dst_size, 2);
const uint32_t exp_tentative_qktv_h = ttnn::transformer::sdpa::streaming_qktv_h(
    exp_tentative_out_h, exp_tentative_out_w, dst_size, Sq_chunk_t);
const bool exp_has_qktv_remainder = (Sq_chunk_t % exp_tentative_qktv_h != 0);

const bool use_streaming_compute =
    !fp32_dest_acc_en && qk_out_subblock_h <= 2 &&
    Sk_chunk_t % qk_out_subblock_w == 0 && qk_in0_num_subblocks > 1 &&
    !exp_has_qktv_remainder;
```

Result: **3/3 PASS**. ⇒ Bug is in the remainder path, **not** in the
larger-num_subblocks Phase 1 path (which still runs in this experiment).

### Experiment 2 — un-hoist drain sub_exp

In the hoisted-drain branch (`compute_streaming.hpp:1146-1174`), moved
the drain `sub_exp_first_col_blocks` + `cb_push_back(cb_exp_max_diff)`
from BEFORE the first SALAD to AFTER the first SALAD + its push_back.

Original order:
1. `cb_push_back(cb_exp_max_diff, drain_h)` ← drain sub_exp's pack
2. `salad_correct_row(salad_row, w_salad, qktv_h)` ← first SALAD's UNPACK
3. `cb_push_back(cur.sum, qktv_h); cb_push_back(out_cb, qktv_h*vDHt)`
4. `salad_correct_row(drain_salad_row, drain_w, drain_h)`

Reordered to:
1. first SALAD's UNPACK
2. push_back cur.sum + out_cb
3. drain sub_exp's pack + push_back
4. drain SALAD

Result on first build: **3/3 PASS**.

### Experiment 3 — explicit PACK→UNPACK barrier

Kept the original hoisted order but added a STALL_SYNC+WAIT_PACK
semaphore handshake immediately after the drain `cb_push_back(cb_exp_max_diff)`:

```cpp
PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
```

Result on first build: **3/3 PASS**.

### Falsified — not a CB-wraparound issue

`cb_exp_max_diff` is sized `statistics_tiles = Sq_chunk_t` (= 3 for Sq=3),
and exactly 3 tiles are pushed per Phase 2 iteration. Doubled the
allocation to rule out wraparound:

```cpp
// ring_joint_sdpa_program_factory.cpp:758
const uint32_t cb_exp_max_diff =
    allocate_tile_cb(2 * statistics_tiles, stats_tile_size, stats_df);
```

Result: **3/3 FAIL**. ⇒ Not a CB sizing/wraparound bug.

## What's unresolved

Each of experiments 1/2/3 passed 3/3 on **first build after the change**,
but later re-runs (after toggling between fixes and clearing
`~/.cache/tt-metal-cache/`) showed 5/5 FAIL with the **same source** that
had passed earlier. Either:

- (a) The 3/3 passes were timing flukes (race-wins-by-luck), or
- (b) The JIT kernel cache is doing something subtle (loading a stale
  binary despite source change), or
- (c) Some other state I'm not accounting for.

The localization (experiment 1) is robust regardless — disabling
streaming for the remainder branch eliminates the race deterministically
because it skips the whole path. The narrower fixes (experiments 2, 3)
need bigger sample sizes to validate.

## Where the race lives — narrowed picture

For Sq=3 partial-K iter the order is:

| step | who   | what                                                              |
|------|-------|-------------------------------------------------------------------|
| A    | PACK  | sub_exp(salad_row=0, qktv_h=2) → cb_exp_max_diff[0,1]              |
| B    | PACK  | V matmul for remainder row → out_cb[row 2]                         |
| C    | PACK  | drain sub_exp(drain_salad_row=2, drain_h=1) → cb_exp_max_diff[2]   |
| D    | UNPACK| first salad reads cb_exp_max_diff[0,1] + prev.out + RMW out_cb[0,1]|
| E    | PACK  | push_back cur.sum, out_cb (advance wr_ptr)                         |
| F    | UNPACK| drain salad reads cb_exp_max_diff[2] + prev.out + RMW out_cb[row 2]|

All three working experiments break the C→D adjacency. Reads at D target
positions 0,1 which were written at A (long flushed); writes at C target
position 2 which is read later at F. Adjacent in time but disjoint in L1.
The fact that an explicit STALL_SYNC+WAIT_PACK barrier between C and D
fixes it suggests `cb_push_back`'s internal `TTI_STALLWAIT(STALL_THCON,
PACK)` is not a strong enough drain for the immediately-following
in-kernel UNPACK — but this needs validation with bigger sample sizes.

## Suggested next experiments

1. **Robustness**: re-run experiment 2 (un-hoist) 20× with cache cleared
   between groups to see if PASSes are reproducible or flukes.
2. **Stale-data hypothesis** (from the original report — not yet tested):
   memset/zero-fill `cb_qkt_im[*, active_Sk..Sk_chunk_t)` before the
   partial-K mask l1-acc in Phase 1. If determinism returns, the root
   cause is mask-onto-stale-after-sub_exp, not Phase-2 ordering.
3. **Bisect**: insert the barrier from experiment 3 in different positions
   (before drain sub_exp, after first salad pop, around the V matmul) to
   triangulate which PACK is racing with which UNPACK.

## Follow-up experiments (2026-05-27, second pass)

Each ran the existing repro test **once** on the same BH quietbox; baseline
re-confirmed at 43,001 diffs / max 0.125 before starting.

| Experiment                                                              | Result | iter / chunk | # diffs | max diff |
|-------------------------------------------------------------------------|--------|--------------|--------:|---------:|
| Barrier at **C→D** (PACK_DONE handshake after drain sub_exp push)       | FAIL   | 1 / 6        |  39,160 |    0.125 |
| Barrier at **E→F** (PACK_DONE handshake before drain SALAD)             | FAIL   | 1 / 6        |  42,105 |    0.125 |
| **Both** barriers together                                              | FAIL   | 1 / 6        |  42,361 |    0.125 |

Each barrier was the same 3-instruction sequence as experiment 3 in this
report (`PACK_DONE` post + UNPACK wait + UNPACK get).

### What this rules out

- The original experiment-3 barrier (C→D) does **not** fix the race on a
  single-run sample. Combined with the MD's prior observation that
  experiment 3's 3/3 PASS was followed by 5/5 FAIL, the most likely
  explanation is that experiments 2 and 3 were **timing flukes**, not
  fixes. The race is not a narrow PACK→UNPACK on `cb_exp_max_diff`.
- The alternate "E→F" hypothesis (drain SALAD's l1-acc PACK on
  `out_cb[row drain_salad_row]` races with the remainder V matmul's
  prior PACK to the same row, with `cb_push_back(out_cb, qktv_h*vDHt)`'s
  `STALL_THCON+PACK` insufficient to drain the PACK write buffer) is
  also **not** the culprit — adding a strong PACK→UNPACK barrier there
  does not change the race signature.
- Combining both barriers is also not a fix → the race is not the union
  of two narrow PACK adjacencies.

### Where to look next

The race lives somewhere outside the two adjacencies inspected so far.
Untested hypotheses in priority order:

1. **Phase 1 stale-data** (already in MD's #2 above) — still untested and
   the cheapest experiment to do next. Force-zero
   `cb_qkt_im[*, active_Sk..Sk_chunk_t)` at start of Phase 1 on partial-K
   iters before the mask l1-acc.
2. **Phase 2 q_sub=0 split-drain** — that block does an in-place sub_exp
   on the last cb_qkt_im row (PACK writes via `pack_tile<true>`) while
   the V matmul UNPACK reads `cb_qkt_im[rows 0..qktv_h-1]` from the
   same K-chunk. PACK→UNPACK on different rows of the same CB; the
   existing `q_num_subblocks == 1` branch has a PACK_DONE handshake
   there (compute_streaming.hpp:976-980), but for our config
   `q_num_subblocks == 3` so that branch doesn't fire.
3. **prev/cur swap state** carrying PACK config (`l1_acc`,
   `pack_relu`, pack format) from the full-K iter into the partial-K
   iter. `sub_exp_block_bcast_cols` resets `l1_acc=0` + `NO_RELU` at end
   (line 367-368) but other paths may leak state.

`git diff HEAD` is empty after revert; no candidate fix is committed.

## Code changes currently committed on this branch

None of the candidate fixes from this investigation are in the source.
Only the existing repro test + bug-report MD remain. `git diff HEAD` is
empty after revert.

## Note for triage

This is a latent bug exposed by the new chunked-prefill determinism test
added in commit `76424224e` on this branch. The bug exists on `main`
(PR #44929, `5b9b5b73bf7`) — it's not introduced by this branch. The
test makes it observable.
