"""Explore arithmetic in the reported v2 table; no model or sample reanalysis.

Outputs are explicitly summary-derived exploratory calculations. This does not
recover HAIC predictions, estimate uncertainty, or execute pending diagnostics.
Existing output is refused; use --output for an independently named rerun.
"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'abstract-20260919-exploratory-summary.json')
    args = parser.parse_args()
    source = root.parent/'proposal/README.md'
    content = source.read_bytes()
    rows = {}
    for line in content.decode().splitlines():
        if line.startswith('| '):
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if len(cells) == 4 and cells[0] in {
                'Unchanged / filter0', 'Coordinate', 'Initialized + readout', 'Paired JEPA', 'Shuffled JEPA', 'Direct'
            }:
                rows[cells[0]] = [Decimal(value) for value in cells[1:]]
    if len(rows) != 6:
        raise ValueError('Expected six declared methods in the reported three-extractor table.')
    result = []
    for i, extractor in enumerate(('hrnet_w32', 'rtmpose_m', 'vitpose_base')):
        raw, coordinate, initialized, paired, shuffled, direct = (
            rows[name][i] for name in ('Unchanged / filter0', 'Coordinate', 'Initialized + readout',
                                       'Paired JEPA', 'Shuffled JEPA', 'Direct'))
        reduction = lambda base, candidate: float(100*(base-candidate)/base)
        result.append(dict(
            extractor=extractor,
            reported_errors={name: str(values[i]) for name, values in rows.items()},
            paired_reduction_vs_unchanged_percent=reduction(raw, paired),
            paired_reduction_vs_coordinate_percent=reduction(coordinate, paired),
            paired_reduction_vs_initialized_percent=reduction(initialized, paired),
            paired_reduction_vs_direct_percent=reduction(direct, paired),
            paired_reduction_vs_shuffled_percent=reduction(shuffled, paired),
            initialized_gain_as_percent_of_paired_gain=float(100*(raw-initialized)/(raw-paired)),
        ))
    payload = dict(
        analysis_status='exploratory_summary_arithmetic',
        data_status='user_reported_HAIC_summary_not_locally_reconstructed_predictions',
        source='proposal/README.md',
        source_sha256=hashlib.sha256(content).hexdigest(),
        independent_development_people=2,
        training_seeds=[17],
        correlated_evaluation_records_per_method=48,
        results=result,
        limitations=[
            'Rounded aggregate input; final digits can differ from prediction-derived contrasts.',
            'No new observations, inference, uncertainty estimates or calibration fits.',
            'Initialized encoder still has a trained nonlinear temporal readout; its gain is not a calibration result.',
            'Gain fractions are descriptive ratios, not explained variance or causal mechanism estimates.',
            'No amplitude-error values or reference-self timing results are available in the input.',
        ],
    )
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
