#!/usr/bin/env python3
"""Export only hybrid model/calibration assets; never bundles the MVTec image dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-dir', required=True, help='Root folder containing models/ and assets/')
    ap.add_argument('--output', default='MVTec_Video_Inspector_Model_Bundle.zip')
    args = ap.parse_args()

    root = Path(args.project_dir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not (root / 'models').is_dir() or not (root / 'assets').is_dir():
        raise SystemExit(f'Expected {root}/models and {root}/assets')

    required = [
        'models/yolo26n.pt',
        'models/wide_resnet50_2_patch_extractor.pth',
        'models/stfpm_teacher_resnet18_grid.pth',
        'models/stfpm_student_resnet18_grid.pth',
        'models/causal_temporal_transformer.pt',
        'models/temporal_transformer_input_scaler.joblib',
        'models/ppo_inspection_policy.zip',
        'assets/stfpm_calibration_grid.npz',
    ]
    for cat in ['grid', 'zipper', 'hazelnut', 'leather', 'toothbrush']:
        required.extend([
            f'models/yolo26_embedding_knn_{cat}.joblib',
            f'models/patchcore_patch_bank_{cat}.pt',
            f'assets/calibration_{cat}.npz',
        ])
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise SystemExit('Required assets are missing:\n' + '\n'.join(missing))

    # Include all models (including optional subtype classifiers/backbone) and only calibration NPZs
    # from assets/. No dataset, feature cache or raw images are copied.
    files = sorted([p for p in (root / 'models').rglob('*') if p.is_file()])
    files += sorted([p for p in (root / 'assets').glob('*calibration*.npz') if p.is_file()])
    manifest = root / 'model_manifest.json'
    if manifest.is_file():
        files.append(manifest)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for p in files:
            zf.write(p, arcname=p.relative_to(root))
    total = sum(p.stat().st_size for p in files)
    print(f'Created: {out}')
    print(f'Files included: {len(files)}')
    print(f'Uncompressed payload: {total / (1024**2):.1f} MiB')
    print('Dataset images, masks, and feature_cache are NOT included.')

if __name__ == '__main__':
    main()
