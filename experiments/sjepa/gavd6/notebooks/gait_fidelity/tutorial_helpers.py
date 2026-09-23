"""Small, transparent adapters used by the Gait Fidelity tutorial notebooks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class Study:
    root: Path
    work: Path
    python: str
    fixture: bool

    def command(self, name: str, *arguments: str, quiet: bool = False) -> None:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root / 'src')
        env['PYTHONNOUSERSITE'] = '1'
        command = [self.python, '-m', 'gavd6_sjepa.research_directions.gait_fidelity',
                   name, '--work', str(self.work), *map(str, arguments)]
        result = subprocess.run(command, cwd=self.root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if not quiet or result.returncode:
            print(result.stdout)
        result.check_returncode()

    def artifact(self, relative: str):
        path = self.work / relative
        if not path.exists():
            raise FileNotFoundError(f'{path} does not exist yet. Complete the preceding stage first.')
        return json.loads(path.read_text()) if path.suffix == '.json' else path

    def bundle_path(self) -> Path:
        ledger = self.artifact('ledger.json')
        try:
            return Path(ledger['completed']['prepare']['result']['bundle'])
        except KeyError as error:
            raise RuntimeError('The shared preparation has not completed. Read status and its worker log.') from error

    def launch(self) -> None:
        if self.fixture:
            self.command('run', '--local')
            return
        command = ['bash', str(self.root / 'slurm/gait-fidelity/run.sh'), 'launch', str(self.work)]
        print(shlex.join(command))
        env = os.environ.copy()
        env['GF_ROOT'] = str(self.root)
        subprocess.run(command, env=env, cwd=self.root, check=True)


def configure(root: Path, work: Path | None = None) -> Study:
    root = root.expanduser().resolve()
    work = (work or Path(os.environ.get('GF_WORK', root / 'outputs/gait-fidelity/tutorial-fixture'))).expanduser().resolve()
    python = os.environ.get('GF_PYTHON', sys.executable)
    config = work / 'config.json'
    if not config.exists():
        if 'GF_WORK' in os.environ or work != root / 'outputs/gait-fidelity/tutorial-fixture':
            raise FileNotFoundError(f'{config} is missing. Run run.sh setup for your chosen study first.')
        temporary = Study(root, work, python, True)
        temporary.command('init', '--fixture', '--root', str(root))
    cfg = json.loads(config.read_text())
    cache = work / 'tutorial-cache'
    (cache / 'matplotlib').mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache))
    study = Study(root, work, python, bool(cfg.get('fixture', False)))
    print(f'Code: {root}\nWork: {work}\nMode: {"software fixture (no scientific evidence)" if study.fixture else "saved source study"}')
    return study


def preview_images(study: Study, limit: int = 6) -> list[Path]:
    """Display retained plots without silently substituting generated examples."""
    from IPython.display import Image, display
    paths = sorted(study.work.rglob('*.png'))[:limit]
    for path in paths:
        print(path.relative_to(study.work))
        display(Image(filename=str(path)))
    if not paths:
        print('No retained PNG images yet. Inspect the preparation viewer once that stage completes.')
    return paths
