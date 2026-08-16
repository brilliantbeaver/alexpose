"""Generate the vector figures for the S-JEPA model internals tutorial."""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = "#f7f5ef"
INK = "#20364e"
BLUE = "#e7f0f8"
GREEN = "#e6f2ea"
WARM = "#f7e7d8"
VIOLET = "#ece6f6"
WHITE = "#ffffff"
ACCENT_BLUE = "#2f6f99"
ACCENT_GREEN = "#5f9e7e"
ACCENT_WARM = "#e07a4b"
ACCENT_VIOLET = "#8a6fb3"
MUTED = "#5c6f84"


def multiline(x, y, lines, css="b", gap=26, anchor="start"):
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else gap
        spans.append(
            f'<tspan x="{x}" dy="{dy}">{escape(str(line))}</tspan>'
        )
    return f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">' + "".join(spans) + "</text>"


def card(x, y, width, height, title, lines=(), fill=WHITE, stroke=INK, title_css="t"):
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.75"/>',
        multiline(x + 24, y + 36, [title], title_css),
    ]
    if lines:
        parts.append(multiline(x + 24, y + 72, lines, "b"))
    return "".join(parts)


def arrow(x1, y1, x2, y2, label=None, dashed=False, color=INK):
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    line = (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
        f'stroke-width="2.5" marker-end="url(#arrow)"{dash}/>'
    )
    if label:
        line += multiline((x1 + x2) / 2, min(y1, y2) - 10, [label], "s", anchor="middle")
    return line


def elbow(points, label=None, dashed=False, color=INK):
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    path = " L ".join(f"{x} {y}" for x, y in points)
    result = (
        f'<path d="M {path}" fill="none" stroke="{color}" stroke-width="2.5" '
        f'marker-end="url(#arrow)"{dash}/>'
    )
    if label:
        x, y = points[len(points) // 2]
        result += multiline(x + 8, y - 10, [label], "s")
    return result


def pill(x, y, width, label, fill=BLUE, stroke=ACCENT_BLUE):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="36" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        + multiline(x + width / 2, y + 24, [label], "s strong", anchor="middle")
    )


def token_grid(x, y, rows, cols, cell=30, gap=6, states=None):
    states = states or {}
    fills = {
        "visible": BLUE,
        "target": WARM,
        "invalid": "#d8e0e7",
        "eligible": GREEN,
        "mask": VIOLET,
    }
    strokes = {
        "visible": ACCENT_BLUE,
        "target": ACCENT_WARM,
        "invalid": "#8a9bab",
        "eligible": ACCENT_GREEN,
        "mask": ACCENT_VIOLET,
    }
    pieces = []
    for row in range(rows):
        for col in range(cols):
            state = states.get((row, col), "visible")
            pieces.append(
                f'<rect x="{x + col * (cell + gap)}" y="{y + row * (cell + gap)}" '
                f'width="{cell}" height="{cell}" rx="6" fill="{fills[state]}" '
                f'stroke="{strokes[state]}" stroke-width="1.5"/>'
            )
            cx = x + col * (cell + gap) + cell / 2
            cy = y + row * (cell + gap) + cell / 2
            if state == "target":
                pieces.append(f'<path d="M {cx-8} {cy-8} L {cx+8} {cy+8} M {cx+8} {cy-8} L {cx-8} {cy+8}" stroke="{INK}" stroke-width="2.5"/>')
            elif state == "eligible":
                pieces.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="none" stroke="{INK}" stroke-width="2.5"/>')
            elif state == "invalid":
                pieces.append(f'<path d="M {cx-9} {cy+9} L {cx+9} {cy-9}" stroke="{INK}" stroke-width="2.5"/>')
    return "".join(pieces)


def svg(title, description, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L8,4 L0,8 Z" fill="context-stroke"/>
    </marker>
  </defs>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
    .h {{ font-size: 32px; font-weight: 700; }}
    .t {{ font-size: 20px; font-weight: 700; }}
    .b {{ font-size: 16px; fill: #33475e; }}
    .s {{ font-size: 14px; fill: {MUTED}; }}
    .strong {{ font-weight: 700; fill: {INK}; }}
    .mono {{ font-family: Menlo, Consolas, monospace; font-size: 15px; }}
    .w {{ font-size: 16px; font-weight: 700; fill: #ffffff; }}
  </style>
  <rect width="1200" height="720" fill="{PAPER}"/>
  {multiline(48, 58, [title], "h")}
  {body}
</svg>
'''


def write(name, title, description, body):
    path = OUT / name
    content = svg(title, description, body)
    if "\u2014" in content:
        raise ValueError(f"Em dash found in {name}")
    path.write_text(content, encoding="utf-8")


write(
    "01_system_map.svg",
    "S-JEPA learns by matching hidden latent targets",
    "A trainable view encoder and predictor match a slowly updated target encoder at masked joint-time positions.",
    card(48, 118, 240, 150, "Augmented view", ["[B, F, V, C]", "full coordinates + mask"], WARM)
    + card(344, 118, 250, 150, "View encoder", ["Trainable student", "Returns [B, K, D]"], WARM)
    + card(650, 118, 250, 150, "Predictor", ["Restores N positions", "Returns [B, M, D]"], VIOLET)
    + card(956, 118, 196, 150, "Predictions", ["Hidden latent", "vectors"], BLUE)
    + arrow(288, 193, 328, 193)
    + arrow(594, 193, 634, 193)
    + arrow(900, 193, 940, 193)
    + card(48, 400, 240, 150, "Cleaned target", ["Same clip, unmasked", "Not raw ground truth"], GREEN)
    + card(344, 400, 250, 150, "Target encoder", ["No gradients", "Returns [B, N, D]"], GREEN)
    + card(650, 400, 250, 150, "Select mask rows", ["Boolean gather", "Returns [B, M, D]"], GREEN)
    + card(956, 400, 196, 150, "Targets", ["selected teacher", "latent vectors"], GREEN)
    + arrow(288, 475, 328, 475)
    + arrow(594, 475, 634, 475)
    + arrow(900, 475, 940, 475)
    + elbow([(469, 284), (469, 330), (469, 384)], "EMA update", True, ACCENT_GREEN)
    + pill(412, 622, 376, "Loss compares predictions and targets", BLUE, ACCENT_BLUE),
)


write(
    "02_tensor_contract.svg",
    "The tensor symbols used throughout the model",
    "A legend maps the batch, frame, joint, coordinate, segment, token, width, visible, and masked dimensions.",
    card(48, 112, 340, 228, "Raw pose dimensions", ["B  batch size", "F  frames", "V  joints", "C  coordinates per joint"], BLUE)
    + card(430, 112, 340, 228, "Patch dimensions", ["L  frames per segment", "S = F / L  segments", "N = S x V  total tokens", "D  encoder width"], GREEN)
    + card(812, 112, 340, 228, "Mask dimensions", ["M  hidden target tokens", "K = N - M  visible tokens", "P  predictor width", "Current wrapper sets P = D"], VIOLET)
    + card(48, 400, 1104, 176, "Standard 64-frame example", ["F=64, V=33, C=3, L=4, S=16, N=528", "Encoder input [B,64,33,3]  |  visible output [B,K,D]", "Predictor and selected teacher target [B,M,D]"], WHITE)
    + pill(338, 620, 524, "D=96 in the recommended real profile", WARM, ACCENT_WARM),
)


write(
    "03_patchify_walkthrough.svg",
    "Patchify groups one joint across four frames",
    "The encoder reshapes and permutes the input so each token contains one joint over a short time segment.",
    card(48, 122, 220, 158, "Input", ["[B,64,33,3]", "frames, joints, xyz"], BLUE)
    + card(326, 122, 240, 158, "Reshape", ["[B,16,4,33,3]", "make 16 segments"], GREEN)
    + card(624, 122, 240, 158, "Permute", ["[B,16,33,4,3]", "joint before time"], WARM)
    + card(922, 122, 230, 158, "Flatten", ["[B,16,33,12]", "4 frames x 3 coords"], VIOLET)
    + arrow(268, 201, 310, 201)
    + arrow(566, 201, 608, 201)
    + arrow(864, 201, 906, 201)
    + card(48, 368, 1104, 208, "What one patch contains", ["Choose segment 7 and LEFT_ANKLE. Its token starts from 12 numbers:", "(x,y,z) at frame 28, then frames 29, 30, and 31.", "It does not combine multiple joints. A learned Linear(12,D) creates the token."], WHITE)
    + pill(368, 620, 464, "One token = one joint x one time segment", WARM, ACCENT_WARM),
)


write(
    "04_position_embeddings.svg",
    "Time and joint identity are added to every token",
    "Learned time and joint vectors broadcast across a segment by joint grid and are added to patch embeddings.",
    card(48, 112, 286, 180, "Patch embedding", ["content vector", "[B,S,V,D]"], BLUE)
    + card(457, 112, 286, 180, "Time position", ["one vector per segment", "[S,D]"], GREEN)
    + card(866, 112, 286, 180, "Joint position", ["one vector per joint", "[V,D]"], WARM)
    + multiline(382, 218, ["+"], "h", anchor="middle")
    + multiline(791, 218, ["+"], "h", anchor="middle")
    + card(196, 392, 808, 174, "Broadcasted addition", ["token[b,s,v] = patch[b,s,v] + time_pos[s] + joint_pos[v]", "The vectors are added, not concatenated. The width remains D.", "A zero coordinate patch can still become nonzero after bias and position terms."], VIOLET)
    + arrow(600, 308, 600, 376)
    + pill(408, 618, 384, "Shape stays [B,S,V,D]", BLUE, ACCENT_BLUE),
)


write(
    "05_transformer_block.svg",
    "One encoder layer mixes tokens, then transforms each token",
    "A pre-norm transformer encoder layer contains attention and a two-linear-layer feed-forward network with four-times hidden width.",
    card(48, 134, 180, 168, "Input", ["[B,K,D]", "visible tokens"], BLUE)
    + card(292, 112, 250, 214, "Self-attention", ["LayerNorm first", "tokens exchange context", "residual connection"], GREEN)
    + card(606, 112, 310, 214, "MLP / feed-forward", ["LayerNorm first", "Linear D -> 4D", "GELU, then Linear 4D -> D", "residual add"], WARM)
    + card(980, 134, 172, 168, "Output", ["[B,K,D]", "same width"], BLUE)
    + arrow(228, 218, 276, 218)
    + arrow(542, 218, 590, 218)
    + arrow(916, 218, 964, 218)
    + card(180, 408, 840, 166, "Two controls that are easy to confuse", ["dim_feedforward = 4D sets hidden units inside each layer.", "depth sets how many complete transformer encoder layers are stacked.", "For D=96: each MLP is 96 -> 384 -> 96."], WHITE)
    + pill(378, 620, 444, "4x means width, not four layers", WARM, ACCENT_WARM),
)


mask_states = {
    (0, 1): "target", (0, 4): "invalid", (1, 0): "eligible",
    (1, 2): "target", (1, 5): "invalid", (2, 1): "target", (2, 4): "eligible",
}
write(
    "06_mask_semantics.svg",
    "The target mask chooses valid authorized joint-time tokens",
    "The grid uses fill plus symbols: a plain cell is ordinary visible context, an X is a hidden eligible target, a circle is eligible visible context, and a slash is invalid visible context.",
    token_grid(72, 160, 3, 6, cell=52, gap=12, states=mask_states)
    + card(520, 112, 632, 252, "Mask rules", ["Only 12 authorized joints can become targets.", "All L frames in a patch must be originally valid.", "True in target_mask means hidden and predicted.", "True in keep_mask means sent to the view encoder."], WHITE)
    + pill(72, 398, 142, "visible", BLUE, ACCENT_BLUE)
    + pill(230, 398, 142, "target + X", WARM, ACCENT_WARM)
    + pill(72, 450, 142, "eligible + O", GREEN, ACCENT_GREEN)
    + pill(230, 450, 142, "invalid + /", "#d8e0e7", "#8a9bab")
    + card(520, 408, 632, 170, "Common-count example", ["Eligible counts: 180, 160, 150, 175", "M = floor(0.60 x 150) = 90 for every sample", "Equal M gives equal K, which dense batching requires."], VIOLET)
    + multiline(72, 628, ["Targets are eligible. Eligible and invalid unmasked tokens both remain visible context."], "s"),
)


write(
    "07_predictor_scatter.svg",
    "The predictor rebuilds a full token canvas",
    "Visible encoded features are scattered into their original locations among learned mask tokens before full-sequence prediction.",
    card(48, 112, 230, 168, "Visible features", ["[B,K,D]", "from view encoder"], BLUE)
    + card(334, 112, 250, 168, "Project width", ["Linear D -> P", "gives [B,K,P]"], GREEN)
    + card(640, 112, 250, 168, "Full canvas", ["start with N mask tokens", "scatter K visible rows"], VIOLET)
    + card(946, 112, 206, 168, "Add positions", ["time + joint", "[B,N,P]"], WARM)
    + arrow(278, 196, 318, 196)
    + arrow(584, 196, 624, 196)
    + arrow(890, 196, 930, 196)
    + card(170, 394, 330, 164, "Predict all N positions", ["transformer over visible", "features plus mask tokens"], VIOLET)
    + card(700, 394, 330, 164, "Return only M targets", ["Linear P -> D", "Boolean gather gives [B,M,D]"], BLUE)
    + arrow(500, 476, 684, 476, "target_mask gather")
    + elbow([(1048, 296), (1048, 354), (335, 354), (335, 378)])
    + pill(360, 620, 480, "Flattened Boolean order preserves alignment", GREEN, ACCENT_GREEN),
)


write(
    "08_forward_shape_flow.svg",
    "SJEPAGait.forward produces two aligned [B,M,D] tensors",
    "The view path predicts masked positions while the target path selects the matching full-encoder positions without gradients.",
    card(48, 112, 190, 148, "view", ["[B,F,V,C]"], WARM)
    + card(286, 112, 230, 148, "view_encoder", ["keep ~target_mask", "[B,K,D]"], WARM)
    + card(564, 112, 230, 148, "predictor", ["restore N positions", "select M"], VIOLET)
    + card(842, 112, 310, 148, "predicted", ["[B,M,D]", "trainable path"], BLUE)
    + arrow(238, 186, 270, 186)
    + arrow(516, 186, 548, 186)
    + arrow(794, 186, 826, 186)
    + card(48, 398, 190, 148, "target", ["[B,F,V,C]", "cleaned clip"], GREEN)
    + card(286, 398, 230, 148, "target_encoder", ["all N tokens", "[B,N,D]"], GREEN)
    + card(564, 398, 230, 148, "mask select", ["same target_mask", "choose M rows"], GREEN)
    + card(842, 398, 310, 148, "selected", ["[B,M,D]", "no gradient path"], GREEN)
    + arrow(238, 472, 270, 472)
    + arrow(516, 472, 548, 472)
    + arrow(794, 472, 826, 472)
    + pill(336, 620, 528, "Same mask and flatten order align every pair", BLUE, ACCENT_BLUE),
)


write(
    "09_gradient_and_ema.svg",
    "Gradients train the student; EMA moves the teacher",
    "Loss gradients pass through the predictor to the view encoder. After the optimizer call, an EMA update is applied from the view encoder down to the target encoder.",
    card(48, 118, 250, 170, "JEPA loss", ["backpropagates through", "predictions only"], BLUE)
    + card(388, 118, 250, 170, "Predictor", ["has gradients", "optimizer updates"], VIOLET)
    + card(728, 118, 250, 170, "View encoder", ["has gradients", "optimizer updates"], WARM)
    + arrow(298, 203, 372, 203, "gradient")
    + arrow(638, 203, 712, 203, "gradient")
    + card(728, 416, 250, 170, "Target encoder", ["requires_grad=False", "updated in no_grad"], GREEN)
    + elbow([(853, 304), (853, 400)], "EMA", True, ACCENT_GREEN)
    + card(48, 416, 590, 170, "EMA rule", ["target = m x target + (1-m) x view", "m rises toward 1 on a cosine schedule", "the teacher changes more slowly over time"], WHITE)
    + pill(156, 620, 888, "No target gradient does not mean frozen state: EMA still mutates its parameters", WARM, ACCENT_WARM),
)


write(
    "10_center_temperature.svg",
    "The JEPA loss compares distributions over latent dimensions",
    "Teacher and predictor vectors are separate aligned inputs to cross-entropy. The teacher is centered and sharpened, while the predictor keeps the gradient path.",
    card(48, 112, 300, 170, "Teacher target", ["subtract target_center", "divide by 0.06", "softmax over D"], GREEN)
    + card(48, 400, 300, 170, "Predictor output", ["divide by 0.10", "log_softmax over D", "keep gradient path"], VIOLET)
    + card(520, 210, 300, 220, "Cross-entropy", ["two aligned inputs", "sum over D", "mean over B and M", "one scalar loss"], BLUE)
    + elbow([(348, 197), (450, 197), (450, 278), (504, 278)])
    + elbow([(348, 485), (450, 485), (450, 362), (504, 362)])
    + card(852, 112, 300, 170, "Center update", ["after the loss", "mean selected targets", "updates next-batch center"], GREEN)
    + card(852, 400, 300, 170, "What D does not mean", ["D components are learned", "latent features, not classes."], WARM)
    + pill(328, 620, 544, "This is latent prediction, not classification", VIOLET, ACCENT_VIOLET),
)


write(
    "11_objective_stack.svg",
    "Three objectives shape the representation",
    "The total training objective combines latent prediction, two-view anti-collapse regularization, and a label-aware group term after Stage 0.",
    card(48, 124, 300, 224, "JEPA", ["masked latent matching", "all curriculum stages", "default weight 1.0"], BLUE)
    + card(450, 124, 300, 224, "VICReg", ["invariance + variance", "+ covariance", "all stages, default outer 0.05"], GREEN)
    + card(852, 124, 300, 224, "Group term", ["within-label compactness", "+ centroid margin", "Stages 1-4, default outer 0.25"], WARM)
    + multiline(398, 260, ["+"], "h", anchor="middle")
    + multiline(800, 260, ["+"], "h", anchor="middle")
    + card(172, 426, 856, 146, "Total", ["L_total = L_JEPA + 0.05 L_VICReg + 0.25 L_group", "L_group is exactly zero when a batch has fewer than two condition labels."], VIOLET)
    + pill(294, 620, 612, "Later stages are label-informed representation learning", WARM, ACCENT_WARM),
)


write(
    "12_pooling_to_384d.svg",
    "Token width D and downstream vector width 4D are different",
    "Target tokens fan out into independent global and authorized validity-aware pools. Their four summaries then fan in by concatenation.",
    card(450, 112, 300, 160, "Target tokens", ["[B,S,V,D]", "D=96 in real profile"], GREEN)
    + card(140, 350, 300, 160, "Global pool", ["valid mean [B,D]", "valid std [B,D]"], BLUE)
    + card(760, 350, 300, 160, "Authorized pool", ["12-joint mean [B,D]", "12-joint std [B,D]"], WARM)
    + elbow([(550, 288), (550, 318), (290, 318), (290, 334)])
    + elbow([(650, 288), (650, 318), (910, 318), (910, 334)])
    + card(260, 558, 680, 108, "Concatenate four D-vectors", ["global mean | global std | authorized mean | authorized std", "[B,4D] = [B,384] when D=96"], VIOLET)
    + elbow([(290, 526), (290, 542), (480, 542)])
    + elbow([(910, 526), (910, 542), (720, 542)]),
)


write(
    "13_preprocessing.svg",
    "Preprocessing separates coordinates from target eligibility",
    "The pipeline cleans and resizes pose coordinates while preserving an original-observation validity mask for masking and pooling.",
    card(48, 112, 190, 180, "Raw pose", ["[T,33,4]", "xyz + visibility"], BLUE)
    + card(286, 112, 220, 180, "Short gaps", ["interpolate <=4 frames", "keep pre-resize validity"], GREEN)
    + card(554, 112, 220, 180, "Normalize", ["pelvis center", "body-width scale"], WARM)
    + card(822, 112, 330, 180, "Resize to F frames", ["coordinates [F,33,3]", "validity [F,33]"], VIOLET)
    + arrow(238, 202, 270, 202)
    + arrow(506, 202, 538, 202)
    + arrow(774, 202, 806, 202)
    + card(48, 404, 510, 170, "Coordinate path", ["long gaps and invalid values become zero sentinels", "zero inputs can still gain bias and position embeddings"], WHITE)
    + card(642, 404, 510, 170, "Validity path", ["resample mask, then threshold at 0.5", "all L resampled flags must be true for a target"], WHITE)
    + pill(276, 620, 648, "The resampled mask controls targets and pooling", GREEN, ACCENT_GREEN),
)


tutorial = Path(__file__).resolve().parent / "sjepa_model_internals.md"
if tutorial.exists() and "\u2014" in tutorial.read_text(encoding="utf-8"):
    raise ValueError(f"Em dash found in {tutorial}")

print(f"wrote 13 SVG figures to {OUT}")
