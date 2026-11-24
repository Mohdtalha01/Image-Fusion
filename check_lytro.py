import pandas as pd
df = pd.read_csv('results/phase4_test_metrics.csv')
lytro = df[df['Dataset'] == 'lytro']
print('Lytro pairs done:', len(lytro['Pair'].unique()))
if len(lytro['Pair'].unique()) > 0:
    print(lytro.groupby('Method')[['QABF', 'NABF', 'MI', 'SSIM', 'VIFP', 'Latency_ms']].mean().round(4))
