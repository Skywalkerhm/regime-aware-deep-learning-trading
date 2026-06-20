import os, sys
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btc_conditional.trainer import BTCTrainer, BTCDataLoader, ALL_FEATS

DATA_PATH = 'data/processed/tsla_full.csv'
OUT_DIR = 'btc_conditional/runs_tsla'
config = {'window': {'seq_len': 30}, 'model': {'dropout': 0.3}, 'trainer': {'lr': 3e-4, 'epochs': 15}}

FOLDS = [
    ('TF 1', '2018-07-01', '2020-07-01'),
    ('TF 2', '2019-07-01', '2021-07-01'),
    ('TF 3', '2020-07-01', '2022-07-01'),
    ('TF 4', '2021-07-01', '2023-07-01'),
    ('TF 5', '2022-01-01', '2024-01-01'),
    ('TF 6', '2022-07-01', '2024-07-01'),
    ('TF 7', '2023-01-01', '2025-01-01'),
    ('TF 8', '2023-07-01', '2025-07-01'),
]

print('TSLA 8-fold LSTM training')
print('=' * 60)
for fname, tr_s, tr_e in FOLDS:
    print(f'{fname}: {tr_s} to {tr_e}')
    dl = BTCDataLoader(DATA_PATH, feats=ALL_FEATS, target_filter=0.005)
    trainer = BTCTrainer(config, feats=ALL_FEATS, label_smoothing=0.1, use_focal=True, focal_gamma=2.0)
    trainer.run(dl, tr_s, tr_e, n_epochs=config['trainer']['epochs'], out_dir=OUT_DIR)
    print(f'  {fname} complete')
print('TSLA training complete')
