import pandas as pd
df = pd.read_csv('results/phase4_test_metrics.csv')

cadm = df[df['Method'] == 'CADM_MSWN_v2']
for ds in cadm['Dataset'].unique():
    sub = cadm[cadm['Dataset'] == ds].sort_values('QABF')
    n = len(sub)
    worst = sub.iloc[0]
    median = sub.iloc[n // 2]
    best = sub.iloc[-1]
    print('===', ds, f'(N={n})', '===')
    print('  Worst  case:', worst['Pair'], f"QABF={worst['QABF']:.4f}, NABF={worst['NABF']:.4f}, SSIM={worst['SSIM']:.4f}")
    print('  Median case:', median['Pair'], f"QABF={median['QABF']:.4f}, NABF={median['NABF']:.4f}, SSIM={median['SSIM']:.4f}")
    print('  Best   case:', best['Pair'], f"QABF={best['QABF']:.4f}, NABF={best['NABF']:.4f}, SSIM={best['SSIM']:.4f}")
