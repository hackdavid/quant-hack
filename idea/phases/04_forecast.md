# Phase 4 — Forecast Agent

**Goal:** a probabilistic 5–60 minute directional forecaster combining
**Kronos** (frozen, used as one expert) with a **custom small TCN** trained
from scratch on crypto + microstructure features, with **meta-labeling**
and **calibration**. Output is a probability distribution over directional
moves, not a point estimate.

**Why fourth:** with a sim ready (Phase 3), we can finally measure
forecast quality in PnL terms, not just statistical metrics.

**Estimated effort:** 7–10 days.

**Activates dep group:** `phase4` (torch, peft, transformers, safetensors,
scikit-learn).

---

## 1. Inputs / outputs

- **Inputs:**
  - Phase 2 features (`state_5m` + `micro_event`).
  - Phase 1 raw klines for Kronos input.
  - GitHub: `https://github.com/shiyu-coder/Kronos` (model weights via HF).
- **Outputs:**
  - `models/forecast/v{N}/` containing:
    - `kronos_lora.safetensors` (LoRA adapter for Kronos)
    - `tcn.safetensors` (custom TCN encoder)
    - `head.safetensors` (joint forecast head)
    - `meta_label.lgbm` (LightGBM meta-label classifier — kept here, not
      Phase 6, because it's tightly coupled to forecast output)
    - `calibrator.pkl` (isotonic regression)
    - `metadata.json`
  - `intraday train forecast` CLI works (pretrain + finetune modes).
  - `intraday predict forecast --at <ts>` returns a `ForecastOutput`.
  - **OOS Brier score < single-feature baseline** (acceptance criterion).

---

## 2. Files to create

```
src/intraday/forecast/
  __init__.py
  output.py                   # ForecastOutput dataclass
  kronos_loader.py            # load Kronos weights, freeze, attach LoRA
  tcn.py                      # small temporal CNN encoder
  head.py                     # joint forecast head
  meta_label.py               # secondary classifier (when to act)
  calibration.py              # isotonic + Platt
  labels.py                   # triple-barrier labels (Lopez de Prado)
  dataset.py                  # PyTorch dataset over Phase 2 features
  splits.py                   # purged k-fold + embargo
  train.py                    # training loop
  predict.py                  # inference path
  cli.py
  configs/
    pretrain.yaml
    finetune.yaml
tests/phase_04/
  test_labels.py
  test_splits.py
  test_kronos_loader.py
  test_tcn.py
  test_head.py
  test_calibration.py
  test_meta_label.py
  test_predict_smoke.py
```

---

## 3. Forecast output schema

```python
class ForecastOutput(BaseModel):
    ts_ms: int
    horizon_minutes: int               # 5 | 15 | 60

    # Probability distribution over signed moves (in vol-normalized units):
    # bins: [-3, -2, -1, -0.5, -0.2, +0.2, +0.5, +1, +2, +3] σ
    p_bins: list[float]                 # len 11, sums to 1.0

    # Convenience scalars (derived):
    p_up_05sigma: float                 # P(move > 0.5σ)
    p_down_05sigma: float
    expected_move_sigma: float          # E[move / σ]
    confidence: float                   # 1 - entropy(p_bins) / max_entropy

    # Meta-label gate:
    meta_act: bool                      # secondary model: should we act?
    meta_p_correct: float               # confidence in primary signal

    # Provenance:
    model_version: str
    inference_ms: float
```

The aggregator (Phase 6) consumes this; the execution policy (Phase 7)
also reads it.

---

## 4. Architecture

```
   ┌──────────────────┐    ┌───────────────────────┐
   │ 1m K-line window │    │ Phase-2 state_5m row  │
   │  (last 256 bars) │    │ + micro_event recent  │
   └────────┬─────────┘    └────────────┬──────────┘
            │                           │
            ▼                           ▼
   ┌──────────────────┐         ┌───────────────┐
   │  Kronos (frozen) │         │  Small TCN    │
   │   + LoRA r=8     │         │ (4 layers,    │
   │   embedding[-2]  │         │  64 channels) │
   └────────┬─────────┘         └────────┬──────┘
            │  256-d                     │  64-d
            └────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Concat + 2-layer   │
              │  MLP forecast head  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Softmax over 11     │
              │ vol-normalized bins │
              └──────────┬──────────┘
                         ▼
                Calibration (isotonic)
                         ▼
                 ForecastOutput
```

A **separate meta-label classifier** (LightGBM) takes the primary
forecast's confidence + a few additional features and predicts whether
the primary's directional signal is correct. This is the López de Prado
meta-labeling pattern: it lets the system *abstain* when confidence is
low.

---

## 5. Function / class signatures

### `src/intraday/forecast/labels.py` — triple-barrier labels

```python
def triple_barrier_labels(
    bars: pl.DataFrame,
    *,
    pt_sl: tuple[float, float],     # profit / stop in σ multiples
    horizon_minutes: int,
    vol_window_minutes: int = 60,
) -> pl.DataFrame:
    """Lopez de Prado 2018 ch.3.

    Returns columns:
        - label_sign: int in {-1, 0, +1}
        - label_first_touch_ms: int
        - label_realized_return: float
    """
```

This is the *only* labeling method allowed. Fixed-time labels are
known-bad on noisy financial data.

### `src/intraday/forecast/splits.py` — purged k-fold + embargo

```python
def purged_kfold(
    timestamps: pl.Series,
    label_first_touch: pl.Series,
    n_splits: int,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Lopez de Prado 2018 ch.7."""
```

### `src/intraday/forecast/kronos_loader.py`

```python
def load_kronos_with_lora(
    *,
    base_checkpoint: Path,           # downloaded from HF
    lora_rank: int = 8,
    lora_target_layers: list[int] = [8, 9, 10, 11],
    freeze_base: bool = True,
) -> tuple[nn.Module, KronosConfig]:
    """Returns Kronos with LoRA adapters injected on the specified
    transformer layers. All non-LoRA params have requires_grad=False.
    """

def kronos_embed(
    model: nn.Module,
    klines_window: torch.Tensor,     # [B, T=256, 5]  (OHLCV)
) -> torch.Tensor:
    """Forward pass returning hidden_states[-2] mean-pooled over time
    → [B, hidden_dim] embedding.
    """
```

### `src/intraday/forecast/tcn.py`

```python
class SmallTCN(nn.Module):
    """4 dilated 1D-conv layers, 64 channels, kernel 3, dropout 0.1.

    Input:  [B, T=128, n_features]  from Phase 2 state_5m
    Output: [B, 64]                  pooled embedding
    """
```

### `src/intraday/forecast/head.py`

```python
class ForecastHead(nn.Module):
    """Concatenates Kronos + TCN embeddings, MLP → logits over 11 bins."""

    def forward(
        self,
        kronos_emb: torch.Tensor,  # [B, hidden]
        tcn_emb: torch.Tensor,     # [B, 64]
    ) -> torch.Tensor:
        # returns logits [B, 11]
```

### `src/intraday/forecast/meta_label.py`

```python
class MetaLabelClassifier:
    """LightGBM classifier: P(primary forecast direction is correct)."""

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None: ...
    def predict_proba(self, X: pl.DataFrame) -> np.ndarray: ...
```

Inputs: forecast confidence, regime hint, vol regime, basis-z, funding-z,
hour-of-day. Output: P(direction correct).

### `src/intraday/forecast/calibration.py`

```python
class IsotonicCalibrator:
    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> None: ...
    def transform(self, p_raw: np.ndarray) -> np.ndarray: ...
```

Fit on validation set predictions to map raw probabilities → calibrated
probabilities. Diagnose with **reliability plot** (saved as PNG in run dir).

### `src/intraday/forecast/predict.py`

```python
def load_forecast(version: str = "latest") -> ForecastModel: ...

class ForecastModel:
    def predict(
        self,
        *,
        klines_window: pl.DataFrame,  # last 256 1m klines
        state_window: pl.DataFrame,   # last 128 5m state rows
        ts_ms: int,
        horizon_minutes: int = 15,
    ) -> ForecastOutput: ...
```

---

## 6. Training plan

### Pretrain (one-time)

1. **Data window:** 12 months of historical (2024-01-01 to 2024-12-31).
2. **Train / val / OOS:**
   - Train: 2024-01 → 2024-09  (9 months)
   - Val:   2024-10 → 2024-11  (2 months, used for early stopping + calibration)
   - OOS:   2024-12             (1 month, never touched until final eval)
3. **Splits inside train:** 5-fold purged k-fold with 1% embargo (~9h).
4. **Loss:** cross-entropy over 11 bins + 0.1 × focal-loss to weight tails.
5. **Optimizer:** AdamW, lr=2e-4 for LoRA + TCN + head, weight_decay=1e-2.
6. **Schedule:** cosine, 5 epochs, batch 256.
7. **Early stopping:** val Brier score, patience 2 epochs.
8. **After best-val checkpoint:**
   - Fit `IsotonicCalibrator` on val predictions.
   - Train `MetaLabelClassifier` on val (out-of-fold predictions to avoid
     leakage): X = primary forecast confidence + side features,
     y = `1[primary direction matches realized triple-barrier label]`.
9. **Save** to `models/forecast/v1/`.

### Fine-tune (monthly, Phase 9)

Same pipeline but:
- Load `models/forecast/v{N}/` as init.
- LoRA-only updates; TCN frozen unless drift on TCN-relevant features
  is detected.
- 30 days of data; 1 epoch; lr halved; **EWC penalty** anchoring
  important weights (Fisher matrix computed on the previous month's
  data).
- Save as `v{N+1}` and run canary in Phase 9.

---

## 7. CLI commands wired

| Command | Behavior |
|---|---|
| `intraday train forecast --mode pretrain --train-start ... --train-end ... --val-end ...` | full pretrain |
| `intraday train forecast --mode finetune --from-version v3 --since 30d` | LoRA-only finetune |
| `intraday predict forecast --at "2024-06-15T10:00:00Z" --horizon 15` | single inference (debug) |
| `intraday backtest run --strategy v2_forecast_only --start ... --end ...` | smoke-test strategy that trades on forecast direction alone (sized 1-fixed) |

---

## 8. Strategy `v2_forecast_only` for validation

Lives in `src/intraday/sim/strategies/v2_forecast_only.py`. Logic:
- At each 5m boundary, call `ForecastModel.predict`.
- If `meta_act and p_up_05sigma > 0.6`, send post-only buy at microprice.
- If `meta_act and p_down_05sigma > 0.6`, send post-only sell.
- Fixed size = 0.01 BTC. Hold 15 min. Exit at horizon or stop-loss = 1σ.

This strategy's **OOS Sharpe is the acceptance criterion** for Phase 4.

---

## 9. Unit tests

### `test_labels.py`
- Synthetic series with known barrier touches → labels match expectation.
- pt_sl asymmetric → asymmetric label distribution.
- Embargo prevents leakage: a label whose `first_touch` lies inside a
  test fold cannot have its features used in train.

### `test_splits.py`
- 5-fold purged: every train sample's first-touch is < every test
  sample's start - embargo.

### `test_kronos_loader.py` (use `pytest.mark.gpu` if needed)
- Load checkpoint, attach LoRA, forward 1 sample, output shape correct.
- Trainable params count = LoRA params only (when `freeze_base=True`).

### `test_tcn.py`
- Forward random tensor, output shape `[B, 64]`.
- Gradient flows.

### `test_head.py`
- Logits shape `[B, 11]`.
- Softmax sums to 1.

### `test_calibration.py`
- Synthetic miscalibrated predictions (`p_pred = 0.7` for events with
  true rate 0.5) → calibrator maps 0.7 → 0.5 within ε.

### `test_meta_label.py`
- Train on synthetic where high confidence → correct, low → wrong;
  AUC > 0.7 on held-out.

### `test_predict_smoke.py`
- End-to-end `ForecastModel.predict` on fixture window returns valid
  `ForecastOutput`.

---

## 10. Integration test

`tests/integration/test_phase_04.py`:

```python
@pytest.mark.slow
def test_pretrain_minimal_runs_end_to_end(tmp_path):
    """1-week toy dataset, 1 epoch, batch 32 → produces a valid v1 model."""
```

```python
@pytest.mark.slow
def test_v2_forecast_only_backtest(tmp_path, pretrained_model):
    """Run v2_forecast_only on the OOS month → metrics.json valid +
    OOS hit rate > 50% (placeholder; real bar is below)."""
```

Manual smoke (after Phase 3 sim and 12 months of data ready):

```bash
# Pretrain (~ a few hours on a single mid-range GPU)
uv run intraday train forecast \
    --mode pretrain \
    --train-start 2024-01-01 --train-end 2024-09-30 \
    --val-start 2024-10-01 --val-end 2024-11-30 \
    --epochs 5 --batch-size 256 --lora-rank 8

# Validate on OOS month
uv run intraday backtest run \
    --strategy v2_forecast_only \
    --start 2024-12-01 --end 2024-12-31 \
    --capital 10000 --report
```

---

## 11. Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | OOS Brier score (Dec 2024) < 0.5 × baseline (where baseline is "always p_up = base rate") | `metrics.json` |
| 2 | OOS reliability plot is monotone, deviation from diagonal at any decile ≤ 0.05 | calibration diagnostic PNG |
| 3 | OOS hit rate when `meta_act=True` strictly > hit rate when `meta_act=False` | inspect log |
| 4 | Total trainable params ≤ 5% of base Kronos params (LoRA + TCN + head) | `metadata.json` |
| 5 | Inference latency p99 < 50 ms on CPU | benchmark |
| 6 | v2_forecast_only **OOS Sharpe ≥ 0.5** on Dec 2024 with realistic costs | backtest report |
| 7 | Re-running pretrain with same seed gives identical val metrics | seeded test |
| 8 | The OOS month is **never** touched during training/val (audit log) | code review |

**Acceptance #6 is the hard gate.** If the forecast can't get OOS
Sharpe ≥ 0.5 on its own (1-fixed sizing, no aggregator), the system
will not get to Sharpe ≥ 1.0 with the aggregator. **Stop and reconsider.**

---

## 12. Common mistakes to avoid

- **Don't fine-tune Kronos base weights.** Always frozen + LoRA.
- **Don't mix label horizon with input horizon.** Input is last 256 1m bars
  (~4h); label horizon is 5–60 minutes forward. Easy bug if not careful.
- **Don't forget the embargo.** Without it, val metrics are ~10–15% better
  than reality.
- **Don't use class weights to "fix" hit rate.** Use focal loss or just
  let calibration handle it.
- **Don't skip the meta-label classifier.** Without it, the primary
  forecast trades on every signal; meta-label is what makes the system
  selective and gets you from Sharpe 0.4 → 0.8 alone.
- **Don't validate with random k-fold.** Always purged k-fold + embargo.
- **Don't trust val metrics over OOS metrics.** Val is for early stopping
  only; OOS is the truth.
- **Kronos may not have a public Python loader yet** — check the GitHub.
  If only a TF/JAX checkpoint exists, write a one-time conversion script
  with hardcoded shape mapping documented inline.

---

## 13. Done ⇒ proceed to `phases/05_other_agents.md`.
