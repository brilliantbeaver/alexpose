"""Cell constructors for the isolated notebooks 15–18."""
from textwrap import dedent
from nbformat.v4 import new_code_cell, new_markdown_cell


def md(text):
    return new_markdown_cell(dedent(text).strip())


def code(text):
    return new_code_cell(dedent(text).strip())


def setup_cell():
    return code('''
        from pathlib import Path
        from dataclasses import asdict, replace
        import os
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        def locate_suite():
            for parent in (Path.cwd(), *Path.cwd().parents):
                for candidate in (parent, parent / "neurips-laterality"):
                    if (candidate / "laterality_extensions/motion_structured_masks.py").is_file():
                        return candidate.resolve()
            raise FileNotFoundError("Run from the research project directory.")

        SUITE_ROOT = locate_suite()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))
        from laterality_extensions.motion_structured_masks import (
            StudyArm, mamp_logits, sample_study_mask, study_arms,
            paired_study_masks, context_cue_audit,
        )
        from laterality_extensions.comparative_masks import motion_scores
        set_matplotlib_formats("svg", "png")
        pd.set_option("display.precision", 3)
        print("Synthetic software demonstration. No new empirical gait result.")
    ''')
