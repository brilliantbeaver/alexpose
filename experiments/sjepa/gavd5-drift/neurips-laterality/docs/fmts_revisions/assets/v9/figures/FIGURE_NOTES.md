# V9 figure production notes

Both figures are drawn at the final 5.5-inch manuscript width using Matplotlib.
PDF embeds TrueType fonts; SVG retains editable text; PNG is a review preview.
STIX serif typography, thin rules, square module outlines, white space and a
single muted blue accent replace the former colored cards and pose icons.
The smallest labels are 8.5 pt; panel titles are 10 pt. Output dimensions and
text-boundary checks are recorded in layout_checks.json.

Figure 1 has three ordered panels: (a) separate measurement and model-input
preparation, (b) label-free pretraining, and (c) frozen source-held-out evaluation.
Solid arrows carry data; the dashed EMA arrow updates teacher weights. The
teacher receives the complete prepared clip, and its targets enter the hidden-
feature loss. The full-view regularizer shares the online encoder. The target y
is used only for the separate ridge readout. Joint/position identities and patch
dimensions are described in the methods. The full-view pool uses
twelve gait joints; the signed target and bilateral readout use five pairs.
There are no schematic participant images or invented motion traces.

Figure 2 reads the 30 per-seed R-squared values and all six means directly from
numerical_evidence.json. Open black circles identify initialization; filled blue
circles identify trained teachers. A short black tick marks each mean; vertical
jitter only separates the five seeds. The latent-feature matching panel displays the recorded
33/75 and 375/375 same-clip diagnostic counts as percentages (44% and 100%).
These checks reuse source videos and models; no binomial interval or independent-
trial interpretation is introduced. Source uncertainty remains in Table 1.
No model fitting, inference, or bootstrap was rerun.
