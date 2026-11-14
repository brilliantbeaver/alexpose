from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'wacv2007-v7.tex').read_text()
s=s.replace('Calibration and transfer to new sites remain untested.', 'We have not tested whether predicted confidence matches observed accuracy, or whether performance transfers to new sites.')
start=s.index('\\section{Conclusion}')
end=s.index('{\\small\n\\bibliographystyle',start)
s=s[:start]+r'''\section{Conclusion}
This study evaluates learned skeleton features in a small collection of gait videos. Across 88 clips from 41 source recordings, S-JEPA and mean pose gave similar scores when recordings received equal weight, while mean pose performed better when clips received equal weight and in four of five test groups. The resampling analysis left the advantage over mean pose unresolved. Future evaluations should retain this simple comparison while testing whether training and frame order add useful information. Establishing a robust clinical measure will also require verified participants and diagnoses, with validation on independent recordings.

'''+s[end:]
(r/'wacv2007-v8.tex').write_text(s)
