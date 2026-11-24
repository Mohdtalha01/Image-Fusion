import os
import yaml
from pathlib import Path

with open('splits.yaml', 'r') as f:
    splits = yaml.safe_load(f)

test_dict = splits.setdefault('test', {})

# 1. TNO (42 test pairs)
tno_dir = Path('data/raw/tno')
if tno_dir.exists():
    tno_pairs = []
    for sub in sorted(tno_dir.iterdir()):
        if sub.is_dir():
            imgs = sorted([p for p in sub.glob('*.*') if p.suffix.lower() in ['.bmp', '.png', '.jpg', '.tif']])
            if len(imgs) >= 2:
                ir_cand, vis_cand = None, None
                for im_p in imgs:
                    nl = im_p.name.lower()
                    if any(x in nl for x in ['ir', 'lwir', 'flir', 'therm']):
                        if ir_cand is None: ir_cand = im_p
                    elif any(x in nl for x in ['vis', 'visual', 'tv', 'day']):
                        if vis_cand is None: vis_cand = im_p
                if ir_cand is None: ir_cand = imgs[0]
                if vis_cand is None: vis_cand = imgs[1] if len(imgs) > 1 else imgs[0]
                tno_pairs.append({'id': f'tno_{sub.name}', 'ir': str(ir_cand), 'vi': str(vis_cand)})
    test_dict['tno'] = tno_pairs[:42]
    print('Populated TNO test pairs:', len(test_dict['tno']))

# 2. RoadScene (220 test pairs, excluding FLIR_00006)
rs_ir = Path('data/raw/roadscene/infrared')
rs_vis = Path('data/raw/roadscene/visible')
if rs_ir.exists() and rs_vis.exists():
    rs_files = sorted(list(rs_ir.glob('*.jpg')))
    rs_pairs = []
    for f in rs_files:
        if 'FLIR_00006' in f.name:
            continue
        vf = rs_vis / f.name
        if vf.exists():
            rs_pairs.append({'id': f'roadscene_{f.stem}', 'ir': str(f), 'vi': str(vf)})
    test_dict['roadscene'] = rs_pairs[:220]
    print('Populated RoadScene test pairs:', len(test_dict['roadscene']))

# 3. MSRS (361 test pairs)
msrs_ir = Path('data/raw/msrs/test/ir')
msrs_vi = Path('data/raw/msrs/test/vi')
if msrs_ir.exists() and msrs_vi.exists():
    ms_files = sorted(list(msrs_ir.glob('*.png')))
    ms_pairs = []
    for f in ms_files:
        vf = msrs_vi / f.name
        if vf.exists():
            ms_pairs.append({'id': f'msrs_test_{f.stem}', 'ir': str(f), 'vi': str(vf)})
    test_dict['msrs'] = ms_pairs[:361]
    print('Populated MSRS test pairs:', len(test_dict['msrs']))

total_test = sum(len(v) for v in test_dict.values())
print('Total test pairs in splits.yaml:', total_test)

with open('splits.yaml', 'w') as f:
    yaml.dump(splits, f, default_flow_style=False, sort_keys=False)
print('Updated splits.yaml successfully.')
