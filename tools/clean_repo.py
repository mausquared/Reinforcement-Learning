"""Repository cleaner helper.
Usage:
  - Dry-run (default): python tools/clean_repo.py
  - Apply moves: python tools/clean_repo.py --apply

It will move analysis scripts and generated figures into an `analysis/` folder,
preserving originals in a git-friendly way.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_SCRIPTS = ROOT / 'analysis' / 'scripts'
ANALYSIS_OUTPUTS = ROOT / 'analysis' / 'outputs'

# Heuristics: candidate analysis scripts (top-level .py not in core modules)
CORE_FILES = {'launcher.py', 'train.py', 'hummingbird_env.py', 'detailed_evaluation.py', 'requirements.txt', 'README.md', 'PROJECT_IMPROVEMENT_GUIDE.md'}

EXT_SCRIPT = {'.py', '.ipynb'}
EXT_OUTPUTS = {'.png', '.jpg', '.jpeg', '.pdf', '.csv', '.json', '.mp4', '.zip', '.pkl', '.npz'}


def find_candidates(root: Path, include_models: bool = False):
    candidates_scripts = []
    candidates_outputs = []
    for p in root.iterdir():
        if p.is_dir():
            continue
        name = p.name
        if name in CORE_FILES or name.startswith('.') or name == Path(__file__).name:
            continue
        ext = p.suffix.lower()
        if ext in EXT_SCRIPT:
            candidates_scripts.append(p)
        elif ext in EXT_OUTPUTS:
            candidates_outputs.append(p)

    # Optionally include outputs under models/ (disabled by default)
    if include_models:
        for sub in (root / 'models').glob('**/*'):
            if sub.is_file() and sub.suffix.lower() in EXT_OUTPUTS:
                candidates_outputs.append(sub)

    for subdir in ('report_plots_individual','report_plots_final','report_plots_candidate','report_plots_v2','report_plots_final_numbered'):
        d = root / subdir
        if d.exists():
            for sub in d.glob('*'):
                if sub.is_file():
                    candidates_outputs.append(sub)
    return sorted(set(candidates_scripts)), sorted(set(candidates_outputs))


def ensure_dirs():
    ANALYSIS_SCRIPTS.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUTS.mkdir(parents=True, exist_ok=True)


def move_files(files, dest, apply=False):
    actions = []
    for f in files:
        rel = f.relative_to(ROOT)
        target = dest / rel.name
        if apply:
            ensure_dirs()
            shutil.move(str(f), str(target))
            actions.append((str(f), str(target)))
        else:
            actions.append((str(f), str(target)))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='actually move files')
    parser.add_argument('--include-models', action='store_true', help='include files under models/ when searching outputs')
    args = parser.parse_args()

    scripts, outputs = find_candidates(ROOT, include_models=args.include_models)

    print('Found candidate analysis scripts:')
    for s in scripts:
        print('  -', s.relative_to(ROOT))
    print('\nFound candidate outputs:')
    for o in outputs[:200]:
        print('  -', o.relative_to(ROOT))
    if len(outputs) > 200:
        print('  ... and', len(outputs)-200, 'more')

    if args.apply:
        print('\nApplying moves...')
        moved = []
        moved.extend(move_files(scripts, ANALYSIS_SCRIPTS, apply=True))
        moved.extend(move_files(outputs, ANALYSIS_OUTPUTS, apply=True))
        print('\nMoved files:')
        for src, dst in moved:
            print('  -', src, '->', dst)
    else:
        print('\nDry run only. Re-run with --apply to move files into analysis/ (scripts -> analysis/scripts, outputs -> analysis/outputs)')

if __name__ == '__main__':
    main()
