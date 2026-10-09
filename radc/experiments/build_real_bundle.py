#!/usr/bin/env python3
"""Build the canonical BTC/TSLA prediction bundle from frozen checkpoints.

The script is intentionally a bridge: it reuses the trained base models but
does not reuse the old backtest or result-selection logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORK_ROOT))

from btc_conditional.model import LSTMModel
from btc_conditional.model_transformer import TransformerModel
from btc_conditional.trainer import BTCDataLoader, BTCTrainer, ALL_FEATS
from src.rutl.data import soft_regime_posterior

BTC_FOLDS = [
    ("fold_1", "2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01", "search"),
    ("fold_2", "2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01", "search"),
    ("fold_3", "2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01", "search"),
    ("fold_4", "2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01", "search"),
    ("fold_5", "2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01", "validation"),
    ("fold_6", "2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01", "validation"),
    ("fold_7", "2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01", "external"),
    ("fold_8", "2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01", "external"),
]
TSLA_FOLDS = [
    ("fold_1", "2018-07-01", "2020-07-01", "2020-07-01", "2021-01-01", "external"),
    ("fold_2", "2019-07-01", "2021-07-01", "2021-07-01", "2022-01-01", "external"),
    ("fold_3", "2020-07-01", "2022-07-01", "2022-07-01", "2023-01-01", "external"),
    ("fold_4", "2021-07-01", "2023-07-01", "2023-07-01", "2024-01-01", "external"),
    ("fold_5", "2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01", "external"),
    ("fold_6", "2022-07-01", "2024-07-01", "2024-07-01", "2025-01-01", "external"),
    ("fold_7", "2023-01-01", "2025-01-01", "2025-01-01", "2025-07-01", "external"),
    ("fold_8", "2023-07-01", "2025-07-01", "2025-07-01", "2026-01-01", "external"),
]


def ecdf_map(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(reference) == 0:
        raise ValueError("empty validation probability array")
    # Average rank for ties; maps to a strict empirical CDF in (0, 1).
    ranks = np.searchsorted(reference, values, side="right")
    return (ranks + 0.5) / (len(reference) + 1.0)


def validation_window(train_start: str, train_end: str) -> tuple[str, str]:
    dates = pd.date_range(train_start, train_end, freq="D")
    cut = int(len(dates) * 0.8)
    return dates[cut].strftime("%Y-%m-%d"), train_end


def trim_sequence(probs: np.ndarray, df: pd.DataFrame, test_start: str, test_end: str) -> tuple[np.ndarray, np.ndarray]:
    mask = (df["datetime"] >= pd.to_datetime(test_start)) & (df["datetime"] < pd.to_datetime(test_end))
    offset = 60 - 30
    n = min(len(probs) - offset, int(mask.sum()) - 30)
    if n < 5:
        raise ValueError("not enough sequence predictions")
    idx = df[mask].index[30:30 + n]
    return probs[offset:offset + n], idx


def build_lstm_fold(
    df: pd.DataFrame,
    fold: tuple,
    data_path: Path,
    model_dir: Path,
    *,
    asset: str = "BTC-USD",
    model_name: str = "lstm",
) -> pd.DataFrame:
    name, tr_s, tr_e, te_s, te_e, split = fold
    ckpt = model_dir / f"train_{tr_s}" / "best_model.pth"
    meta_path = model_dir / f"train_{tr_s}" / "threshold.json"
    if not ckpt.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing LSTM checkpoint or metadata: {ckpt}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dl = BTCDataLoader(str(data_path), feats=ALL_FEATS)
    dl.apply_saved_norm(meta["feat_mean"], meta["feat_std"])
    trainer = BTCTrainer(
        {"window": {"seq_len": 30}, "model": {"dropout": 0.3}, "trainer": {"lr": 3e-4, "epochs": 15}},
        feats=ALL_FEATS,
    )
    trainer.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    trainer.model.eval()

    val_start, val_end = validation_window(tr_s, tr_e)
    val_probs, _, _ = trainer.predict(
        dl,
        (pd.to_datetime(val_start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
        val_end,
    )
    if val_probs is None:
        raise ValueError(f"{name}: no validation predictions")
    # The sequence prediction begins after the 30-day window.
    val_mask = (df["datetime"] >= pd.to_datetime(val_start)) & (df["datetime"] < pd.to_datetime(val_end))
    n_val = min(len(val_probs) - 30, int(val_mask.sum()) - 30)
    if n_val < 5:
        raise ValueError(f"{name}: no usable validation predictions")
    val_p = val_probs[30:30 + n_val]

    test_probs, _, _ = trainer.predict(
        dl,
        (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
        te_e,
    )
    if test_probs is None:
        raise ValueError(f"{name}: no test predictions")
    test_p, idx = trim_sequence(test_probs, df, te_s, te_e)
    test = df.loc[idx].copy()
    test["p"] = test_p
    test["u"] = ecdf_map(val_p, test_p)
    test["asset"] = asset
    test["model"] = model_name
    test["fold"] = name
    test["split"] = split
    test["r_next"] = df["return_1d"].shift(-1).loc[idx].to_numpy()
    test["y"] = df["target"].loc[idx].to_numpy()
    return test


def build_transformer_fold(df: pd.DataFrame, fold: tuple, data_path: Path, model_dir: Path) -> pd.DataFrame:
    name, tr_s, tr_e, te_s, te_e, split = fold
    ckpt = model_dir / f"train_{tr_s}" / "best_model.pth"
    meta_path = model_dir / f"train_{tr_s}" / "threshold.json"
    if not ckpt.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing Transformer checkpoint or metadata: {ckpt}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dl = BTCDataLoader(str(data_path), feats=ALL_FEATS)
    dl.apply_saved_norm(meta["feat_mean"], meta["feat_std"])
    model = TransformerModel(n_features=len(ALL_FEATS)).cpu()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    model.eval()

    val_start, val_end = validation_window(tr_s, tr_e)
    lb = (pd.to_datetime(val_start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    xh, xr, _, _ = dl.get_window_data(lb, val_end, filter_training=False)
    with torch.no_grad():
        val_probs = model(torch.tensor(xh, dtype=torch.float32), torch.tensor(xr, dtype=torch.float32))[:, 0].numpy()
    val_mask = (df["datetime"] >= pd.to_datetime(val_start)) & (df["datetime"] < pd.to_datetime(val_end))
    n_val = min(len(val_probs) - 30, int(val_mask.sum()) - 30)
    if n_val < 5:
        raise ValueError(f"{name}: no usable validation predictions")
    val_p = val_probs[30:30 + n_val]

    lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    xh, xr, _, _ = dl.get_window_data(lb, te_e, filter_training=False)
    with torch.no_grad():
        test_probs = model(torch.tensor(xh, dtype=torch.float32), torch.tensor(xr, dtype=torch.float32))[:, 0].numpy()
    test_p, idx = trim_sequence(test_probs, df, te_s, te_e)
    test = df.loc[idx].copy()
    test["p"] = test_p
    test["u"] = ecdf_map(val_p, test_p)
    test["asset"] = "BTC-USD"
    test["model"] = "transformer"
    test["fold"] = name
    test["split"] = split
    test["r_next"] = df["return_1d"].shift(-1).loc[idx].to_numpy()
    test["y"] = df["target"].loc[idx].to_numpy()
    return test


def build_xgboost_fold(df: pd.DataFrame, fold: tuple) -> pd.DataFrame:
    import xgboost as xgb

    name, tr_s, tr_e, te_s, te_e, split = fold
    train_mask = (df["datetime"] >= pd.to_datetime(tr_s)) & (df["datetime"] < pd.to_datetime(tr_e))
    test_mask = (df["datetime"] >= pd.to_datetime(te_s)) & (df["datetime"] < pd.to_datetime(te_e))
    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    train_x = train[ALL_FEATS].to_numpy(dtype=np.float32)
    test_x = test[ALL_FEATS].to_numpy(dtype=np.float32)
    train_y = train["target"].to_numpy(dtype=int)
    valid_ok = ~np.isnan(train_x).any(axis=1)
    test_ok = ~np.isnan(test_x).any(axis=1)
    train_x, train_y = train_x[valid_ok], train_y[valid_ok]
    test_x, test = test_x[test_ok], test.loc[test_ok].copy()
    split_at = int(len(train_x) * 0.8)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_x, train_y)
    val_p = model.predict_proba(train_x[split_at:])[:, 1]
    raw_test_p = model.predict_proba(test_x)[:, 1]
    test["p"] = raw_test_p
    test["u"] = ecdf_map(val_p, raw_test_p)
    test["asset"] = "BTC-USD"
    test["model"] = "xgboost"
    test["fold"] = name
    test["split"] = split
    test["r_next"] = df["return_1d"].shift(-1).loc[test.index].to_numpy()
    test["y"] = df["target"].loc[test.index].to_numpy()
    return test


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    posterior = soft_regime_posterior(frame)
    frame = frame.copy()
    if "date" not in frame.columns or frame["date"].isna().all():
        if "datetime" not in frame.columns:
            raise KeyError("real frame must contain datetime or date")
        frame["date"] = pd.to_datetime(frame["datetime"])
    frame["posterior_crisis"] = posterior["crisis"].to_numpy()
    frame["posterior_trend_up"] = posterior["trend_up"].to_numpy()
    frame["posterior_trend_down"] = posterior["trend_down"].to_numpy()
    frame["posterior_chop"] = posterior["chop"].to_numpy()
    frame["uncertainty"] = posterior["uncertainty"].to_numpy()
    if "trend_strength" not in frame:
        frame["trend_strength"] = frame["sma_ratio_20_50"].astype(float) - 1.0
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(WORK_ROOT / "artifacts" / "bundles" / "real_v1.csv"))
    parser.add_argument("--btc-data", default=str(PROJECT_ROOT / "data" / "processed" / "btc_daily_full.csv"))
    parser.add_argument("--btc-lstm-dir", default=str(PROJECT_ROOT / "btc_conditional" / "runs_v7"))
    parser.add_argument("--btc-transformer-dir", default=str(PROJECT_ROOT / "btc_conditional" / "runs_transformer"))
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--include-tsla", action="store_true")
    parser.add_argument("--tsla-data", default=str(PROJECT_ROOT / "data" / "processed" / "tsla_full.csv"))
    parser.add_argument("--tsla-lstm-dir", default=str(PROJECT_ROOT.parent / "LSTM量化金融" / "btc_conditional" / "runs_tsla"))
    args = parser.parse_args()

    btc_df = pd.read_csv(args.btc_data, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    frames = []
    for fold in BTC_FOLDS:
        try:
            frame = build_lstm_fold(btc_df, fold, Path(args.btc_data), Path(args.btc_lstm_dir))
            frames.append(add_regime_features(frame))
            print("built", "BTC-USD", "lstm", fold[0], len(frame), flush=True)
        except Exception as exc:
            print("skip", "BTC-USD", "lstm", fold[0], type(exc).__name__, exc, flush=True)
    if not args.skip_transformer:
        for fold in BTC_FOLDS:
            try:
                frame = build_transformer_fold(btc_df, fold, Path(args.btc_data), Path(args.btc_transformer_dir))
                frames.append(add_regime_features(frame))
                print("built", "BTC-USD", "transformer", fold[0], len(frame), flush=True)
            except Exception as exc:
                print("skip", "BTC-USD", "transformer", fold[0], type(exc).__name__, exc, flush=True)
    if not args.skip_xgboost:
        for fold in BTC_FOLDS:
            try:
                frame = build_xgboost_fold(btc_df, fold)
                frames.append(add_regime_features(frame))
                print("built", "BTC-USD", "xgboost", fold[0], len(frame), flush=True)
            except Exception as exc:
                print("skip", "BTC-USD", "xgboost", fold[0], type(exc).__name__, exc, flush=True)

    if args.include_tsla:
        tsla_df = pd.read_csv(args.tsla_data, parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
        for fold in TSLA_FOLDS:
            try:
                frame = build_lstm_fold(
                    tsla_df,
                    fold,
                    Path(args.tsla_data),
                    Path(args.tsla_lstm_dir),
                    asset="TSLA",
                    model_name="lstm",
                )
                frames.append(add_regime_features(frame))
                print("built", "TSLA", "lstm", fold[0], len(frame), flush=True)
            except Exception as exc:
                print("skip", "TSLA", "lstm", fold[0], type(exc).__name__, exc, flush=True)

    if not frames:
        raise SystemExit("no frames built")
    bundle = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(output, index=False)
    print(json.dumps({
        "output": str(output),
        "rows": len(bundle),
        "assets": sorted(bundle["asset"].unique().tolist()),
        "models": sorted(bundle["model"].unique().tolist()),
        "splits": bundle["split"].value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
