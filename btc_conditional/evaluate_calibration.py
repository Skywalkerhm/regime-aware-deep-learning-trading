import os, sys, json
import numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS
from btc_conditional.metrics import compute_calibration
from btc_conditional.model import LSTMModel
from btc_conditional.model_transformer import TransformerModel
import xgboost as xgb

DATA_PATH = 'data/processed/btc_daily_full.csv'
RESULTS_DIR = 'btc_conditional/results_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

FOLDS = [('Fold 1','2018-07-01','2020-07-01','2020-07-01','2021-01-01'),
    ('Fold 2','2019-07-01','2021-07-01','2021-07-01','2022-01-01'),
    ('Fold 3','2020-07-01','2022-07-01','2022-07-01','2023-01-01'),
    ('Fold 4','2021-07-01','2023-07-01','2023-07-01','2024-01-01'),
    ('Fold 5','2022-01-01','2024-01-01','2024-01-01','2024-07-01'),
    ('Fold 6','2022-07-01','2024-07-01','2024-07-01','2025-01-01'),
    ('Fold 7','2023-01-01','2025-01-01','2025-01-01','2025-07-01'),
    ('Fold 8','2023-07-01','2025-07-01','2025-07-01','2026-01-01')]
REGIMES = ['Recov','Peak','Crash','Recov','Bull','Bull','Bull','Bear']
config = {'window':{'seq_len':30},'model':{'dropout':0.3},'trainer':{'lr':3e-4,'epochs':15}}

def load_norm(ckpt_path):
    thr_path = ckpt_path.replace('best_model.pth','threshold.json')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            d = json.load(f)
        return d.get('feat_mean'), d.get('feat_std')
    return None, None

def predict_transformer(dl, te_s, te_e, ckpt_path):
    """Predict using TransformerModel (architecture differs from LSTM)."""
    from btc_conditional.model_transformer import TransformerModel
    model = TransformerModel(n_features=len(ALL_FEATS))
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    thr_path = ckpt_path.replace('best_model.pth', 'threshold.json')
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            td = json.load(f)
        if td.get('feat_mean') is not None:
            dl.apply_saved_norm(td['feat_mean'], td['feat_std'])
    lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    x_hist, x_real, y, ret = dl.get_window_data(lb, te_e, filter_training=False)
    if x_hist is None:
        return None, None
    xh_t = torch.tensor(x_hist, dtype=torch.float32)
    xr_t = torch.tensor(x_real, dtype=torch.float32)
    with torch.no_grad():
        probs = model(xh_t, xr_t)[:, 0].cpu().numpy()
    offset = 60 - 30
    msk = (dl.df['datetime'] >= pd.to_datetime(te_s)) & (dl.df['datetime'] < pd.to_datetime(te_e))
    n = min(len(probs)-offset, msk.sum()-30)
    if n < 5:
        return None, None
    pt = probs[offset:offset+n]
    yt = np.array(y[offset:offset+n]).ravel()
    return pt, yt

def predict_lstm(dl, te_s, te_e, ckpt_path):
    trainer = BTCTrainer(config, feats=ALL_FEATS)
    trainer.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    feat_mean, feat_std = load_norm(ckpt_path)
    if feat_mean is not None:
        dl.apply_saved_norm(feat_mean, feat_std)
    lb = (pd.to_datetime(te_s) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    probs, y, ret = trainer.predict(dl, lb, te_e)
    offset = 60 - 30
    msk = (dl.df['datetime'] >= pd.to_datetime(te_s)) & (dl.df['datetime'] < pd.to_datetime(te_e))
    n = min(len(probs)-offset, msk.sum()-30)
    if n < 5: return None, None
    return probs[offset:offset+n], np.array(y[offset:offset+n]).ravel()

print('CALIBRATION COMPARISON: LSTM vs XGBoost vs Transformer')
print('=' * 70)
results = []

df_full = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
df_full = df_full.sort_values('datetime').reset_index(drop=True)

for idx, (fname, tr_s, tr_e, te_s, te_e) in enumerate(FOLDS):
    print(f'{fname} [{REGIMES[idx]:>5}]:', end=' ')
    msk_te = (df_full['datetime'] >= pd.to_datetime(te_s)) & (df_full['datetime'] < pd.to_datetime(te_e))
    y_test_all = df_full.loc[msk_te, 'target'].values
    
    row = {'fold': fname, 'regime': REGIMES[idx]}
    
    # 1. LSTM
    ckpt_lstm = f'btc_conditional/runs_v7/train_{tr_s[:10]}/best_model.pth'
    dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
    pt, yt = predict_lstm(dl, te_s, te_e, ckpt_lstm)
    if pt is not None:
        cal = compute_calibration(pt, yt)
        row['lstm_ece'], row['lstm_brier'], row['lstm_acc'] = cal['ece'], cal['brier'], (yt == (pt>=0.5).astype(int)).mean()
    
    # 2. XGBoost
    msk_tr = (df_full['datetime'] >= pd.to_datetime(tr_s)) & (df_full['datetime'] < pd.to_datetime(tr_e))
    X_tr = df_full[msk_tr][ALL_FEATS].values.astype(np.float32)
    y_tr = df_full[msk_tr]['target'].values
    X_te = df_full[msk_te][ALL_FEATS].values.astype(np.float32)
    mask_tr = ~np.isnan(X_tr).any(axis=1)
    mask_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te, y_te = X_te[mask_te], y_test_all[mask_te]
    if len(X_tr) > 50 and len(X_te) > 10:
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, objective='binary:logistic', random_state=42)
        xgb_model.fit(X_tr, y_tr)
        xgb_probs = xgb_model.predict_proba(X_te)[:, 1]
        cal = compute_calibration(xgb_probs, y_te)
        row['xgb_ece'], row['xgb_brier'], row['xgb_acc'] = cal['ece'], cal['brier'], (y_te == (xgb_probs>=0.5).astype(int)).mean()
    
    # 3. Transformer (skipped - model loading separate)
    ## 3. Transformer
    ckpt_tf = f'btc_conditional/runs_transformer/train_{tr_s[:10]}/best_model.pth'
    if os.path.exists(ckpt_tf):
        dl2 = BTCDataLoader(DATA_PATH, feats=ALL_FEATS)
        pt2, yt2 = predict_transformer(dl2, te_s, te_e, ckpt_tf)
        if pt2 is not None:
            cal = compute_calibration(pt2, yt2)
            row['tf_ece'], row['tf_brier'], row['tf_acc'] = cal['ece'], cal['brier'], (yt2 == (pt2>=0.5).astype(int)).mean()
    
    print(f'LSTM ece={row.get("lstm_ece",0):.3f} XGB ece={row.get("xgb_ece",0):.3f} TF ece={row.get("tf_ece",0):.3f}')
    results.append(row)

# Summary
print('\n' + '=' * 70)
print('MODEL       ECE     Brier   Acc')
print('-' * 35)
for model in ['lstm', 'xgb', 'tf']:
    ece = np.mean([r[model+'_ece'] for r in results if model+'_ece' in r])
    brier = np.mean([r[model+'_brier'] for r in results if model+'_brier' in r])
    acc = np.mean([r.get(model+'_acc', 0) for r in results if model+'_acc' in r])
    print(f'{model.upper():<10} {ece:.4f}   {brier:.4f}  {acc:.1%}')

# Save
pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/calibration_comparison.csv', index=False)
print(f'\nSaved to {RESULTS_DIR}/calibration_comparison.csv')
