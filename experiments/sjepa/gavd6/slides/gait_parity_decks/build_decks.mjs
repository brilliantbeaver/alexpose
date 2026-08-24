import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/theodoremui/dev/alexpose/experiments/sjepa";
const OUT = path.join(ROOT, "gavd6/slides/gait_parity_decks");
const IMG = path.join(OUT, "images");

const C = {
  navy: "#102A43",
  blue: "#3D8DFF",
  cyan: "#6DCBF4",
  teal: "#2A9D8F",
  orange: "#F4A261",
  red: "#D95D5D",
  ink: "#111827",
  slate: "#52606D",
  line: "#C9D2DC",
  panel: "#F3F6F9",
  paleBlue: "#EAF5FB",
  paleTeal: "#E8F5F2",
  paleOrange: "#FFF3E8",
  paleRed: "#FDECEC",
  white: "#FFFFFF",
};

const SRC = {
  pdf: "/Users/theodoremui/Downloads/GaitParity Study.pdf",
  nb01: path.join(ROOT, "gavd6/notebooks/foundations/01_gavd_manifest_and_youtube.ipynb"),
  nb04: path.join(ROOT, "gavd6/notebooks/foundations/04_pretrain_sjepa_on_normal.ipynb"),
  nb05: path.join(ROOT, "gavd6/notebooks/foundations/05_inspect_latent_motion.ipynb"),
  nb06: path.join(ROOT, "gavd6/notebooks/foundations/06_capstone_health_condition_classifiers.ipynb"),
  late: path.join(ROOT, "gavd6/artifacts/notebook_runs/organized-2026-08-23/experiments/idea05_signed_laterality/01_probe.ipynb"),
  parity: path.join(ROOT, "gavd6/artifacts/notebook_runs/organized-2026-08-23/experiments/idea09_reflection_equivariance/06_cpu_replication.ipynb"),
  method: path.join(ROOT, "gavd6/notes/research/programs/gait_parity/METHODOLOGY.md"),
  long: path.join(ROOT, "gavd6/notes/research/programs/gait_parity/README_LONG_TERM.md"),
  current: path.join(ROOT, "gavd6/notes/research/programs/gait_parity/tutorials/02-current-direction-and-results.md"),
  planned: path.join(ROOT, "gavd6/notes/research/programs/gait_parity/tutorials/03-planned-research-direction.md"),
  mentor: "User-provided research mentor meeting notes, 2026-08-23",
};

function notes(slide, sources, presenter = "") {
  const lines = [];
  if (presenter) lines.push(presenter, "");
  lines.push("[Sources]");
  for (const source of sources) lines.push(`- ${source}`);
  slide.speakerNotes.textFrame.setText(lines.join("\n"));
  slide.speakerNotes.setVisible(true);
}

function addText(slide, text, position, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: opts.fontSize ?? 20,
    typeface: opts.typeface ?? "Arial",
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
    wrap: "square",
  };
  return box;
}

function addBox(slide, position, opts = {}) {
  return slide.shapes.add({
    geometry: opts.geometry ?? "roundRect",
    position,
    fill: opts.fill ?? C.panel,
    line: { style: "solid", fill: opts.line ?? C.line, width: opts.lineWidth ?? 1 },
    borderRadius: opts.borderRadius ?? "rounded-xl",
  });
}

function addRule(slide, left, top, width, color = C.line, height = 2) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addBase(presentation, title, section, page, accent = C.blue) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addRule(slide, 0, 0, 1280, accent, 8);
  addText(slide, section.toUpperCase(), { left: 48, top: 24, width: 480, height: 24 }, { fontSize: 12, color: accent, bold: true });
  addText(slide, title, { left: 48, top: 53, width: 1160, height: 62 }, { fontSize: 38, bold: true, color: C.navy });
  addText(slide, String(page).padStart(2, "0"), { left: 1185, top: 671, width: 48, height: 20 }, { fontSize: 12, color: C.slate, alignment: "right" });
  addRule(slide, 48, 671, 1108, C.line, 1);
  return slide;
}

function addCover(presentation, cfg) {
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  addRule(slide, 0, 0, 1280, cfg.accent, 12);
  addText(slide, cfg.eyebrow.toUpperCase(), { left: 64, top: 58, width: 700, height: 32 }, { fontSize: 15, color: cfg.accent, bold: true });
  addText(slide, cfg.title, { left: 64, top: 148, width: 1050, height: 190 }, { fontSize: 54, color: C.white, bold: true });
  addText(slide, cfg.subtitle, { left: 68, top: 375, width: 980, height: 105 }, { fontSize: 25, color: "#D9E2EC" });
  addBox(slide, { left: 64, top: 543, width: 1152, height: 92 }, { fill: "#173F5F", line: "#335F7F" });
  addText(slide, cfg.bottom, { left: 92, top: 568, width: 1095, height: 45 }, { fontSize: 19, color: C.white, bold: true });
  notes(slide, cfg.sources, cfg.presenter);
}

async function addImage(slide, filename, position, alt) {
  const blob = await fs.readFile(path.join(IMG, filename));
  slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

async function addPhoto(slide, filename, position, alt) {
  const blob = await fs.readFile(path.join(IMG, filename));
  const ext = path.extname(filename).toLowerCase();
  slide.images.add({
    blob,
    contentType: ext === ".png" ? "image/png" : "image/jpeg",
    alt,
    fit: "cover",
    position,
    geometry: "roundRect",
    borderRadius: "rounded-xl",
  });
}

function addBullets(slide, bullets, position, opts = {}) {
  const joined = bullets.map((b) => `• ${b}`).join("\n");
  return addText(slide, joined, position, { fontSize: opts.fontSize ?? 19, color: opts.color ?? C.ink, bold: opts.bold ?? false });
}

function addCard(slide, card, position, accent = C.blue) {
  addBox(slide, position, { fill: card.fill ?? C.panel, line: card.line ?? C.line });
  addRule(slide, position.left, position.top, position.width, card.accent ?? accent, 6);
  if (card.kicker) addText(slide, card.kicker.toUpperCase(), { left: position.left + 22, top: position.top + 22, width: position.width - 44, height: 22 }, { fontSize: 12, color: card.accent ?? accent, bold: true });
  addText(slide, card.title, { left: position.left + 22, top: position.top + (card.kicker ? 54 : 28), width: position.width - 44, height: card.titleHeight ?? 60 }, { fontSize: card.titleSize ?? 26, color: C.navy, bold: true });
  if (card.body) addText(slide, card.body, { left: position.left + 22, top: position.top + (card.kicker ? 120 : 98), width: position.width - 44, height: position.height - (card.kicker ? 140 : 118) }, { fontSize: card.bodySize ?? 17, color: C.slate });
}

function addThreeCards(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.subtitle) addText(slide, cfg.subtitle, { left: 48, top: 119, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const top = cfg.subtitle ? 186 : 145;
  const h = 452;
  const w = 366;
  const gap = 24;
  cfg.cards.forEach((card, i) => addCard(slide, card, { left: 48 + i * (w + gap), top, width: w, height: h }, cfg.accent));
  if (cfg.takeaway) {
    addBox(slide, { left: 48, top: 612, width: 1120, height: 44 }, { fill: C.navy, line: C.navy });
    addText(slide, cfg.takeaway, { left: 68, top: 624, width: 1080, height: 24 }, { fontSize: 16, color: C.white, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

async function addImageSlide(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const imgTop = cfg.lead ? 178 : 132;
  const imgHeight = cfg.callout ? 390 : 485;
  addBox(slide, { left: 48, top: imgTop, width: 1120, height: imgHeight }, { fill: C.white, line: C.line });
  await addImage(slide, cfg.image, { left: 64, top: imgTop + 14, width: 1088, height: imgHeight - 28 }, cfg.alt);
  if (cfg.callout) {
    addBox(slide, { left: 48, top: imgTop + imgHeight + 18, width: 1120, height: 80 }, { fill: cfg.calloutFill ?? C.paleBlue, line: cfg.calloutLine ?? C.cyan });
    addText(slide, cfg.callout, { left: 72, top: imgTop + imgHeight + 37, width: 1072, height: 44 }, { fontSize: 18, color: C.navy, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

function addTwoColumn(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 54 }, { fontSize: 21, color: C.slate });
  const top = cfg.lead ? 188 : 145;
  const h = cfg.takeaway ? 392 : 455;
  addCard(slide, cfg.left, { left: 48, top, width: 545, height: h }, cfg.accent);
  addCard(slide, cfg.right, { left: 623, top, width: 545, height: h }, cfg.accent2 ?? C.teal);
  if (cfg.takeaway) {
    addBox(slide, { left: 48, top: 600, width: 1120, height: 56 }, { fill: C.navy, line: C.navy });
    addText(slide, cfg.takeaway, { left: 70, top: 616, width: 1076, height: 30 }, { fontSize: 18, color: C.white, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

function addPipeline(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const top = 218;
  const n = cfg.steps.length;
  const gap = 34;
  const w = (1120 - gap * (n - 1)) / n;
  cfg.steps.forEach((step, i) => {
    if (i < n - 1) addText(slide, "→", { left: 48 + i * (w + gap) + w + 3, top: top + 82, width: 28, height: 42 }, { fontSize: 28, color: cfg.accent, bold: true, alignment: "center" });
  });
  cfg.steps.forEach((step, i) => {
    const left = 48 + i * (w + gap);
    addBox(slide, { left, top, width: w, height: 255 }, { fill: step.fill ?? C.panel, line: step.line ?? C.line });
    addText(slide, String(i + 1).padStart(2, "0"), { left: left + 18, top: top + 18, width: 40, height: 26 }, { fontSize: 13, color: step.accent ?? cfg.accent, bold: true });
    addText(slide, step.title, { left: left + 18, top: top + 58, width: w - 36, height: 64 }, { fontSize: 22, color: C.navy, bold: true });
    addText(slide, step.body, { left: left + 18, top: top + 132, width: w - 36, height: 98 }, { fontSize: 16, color: C.slate });
  });
  if (cfg.takeaway) {
    addBox(slide, { left: 178, top: 524, width: 864, height: 72 }, { fill: C.paleBlue, line: C.cyan });
    addText(slide, cfg.takeaway, { left: 208, top: 545, width: 804, height: 34 }, { fontSize: 18, color: C.navy, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

function addMetricSlide(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const top = 195;
  const w = 350;
  const gap = 35;
  cfg.metrics.forEach((m, i) => {
    const left = 48 + i * (w + gap);
    addBox(slide, { left, top, width: w, height: 255 }, { fill: m.fill ?? C.panel, line: m.accent ?? cfg.accent });
    addText(slide, m.value, { left: left + 24, top: top + 30, width: w - 48, height: 72 }, { fontSize: 39, color: m.accent ?? cfg.accent, bold: true });
    addText(slide, m.label, { left: left + 24, top: top + 112, width: w - 48, height: 52 }, { fontSize: 20, color: C.navy, bold: true });
    addText(slide, m.note, { left: left + 24, top: top + 174, width: w - 48, height: 55 }, { fontSize: 15, color: C.slate });
  });
  addBox(slide, { left: 48, top: 483, width: 1120, height: 120 }, { fill: cfg.bottomFill ?? C.paleBlue, line: cfg.accent });
  addText(slide, cfg.bottomTitle, { left: 76, top: 508, width: 260, height: 32 }, { fontSize: 18, color: cfg.accent, bold: true });
  addText(slide, cfg.bottomBody, { left: 330, top: 503, width: 810, height: 70 }, { fontSize: 18, color: C.navy, bold: true });
  notes(slide, cfg.sources, cfg.presenter);
}

function addBarResult(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  addBox(slide, { left: 48, top: 184, width: 680, height: 420 }, { fill: C.white, line: C.line });
  slide.charts.add("bar", {
    position: { left: 78, top: 218, width: 620, height: 340 },
    categories: cfg.categories,
    series: cfg.series,
    hasLegend: cfg.series.length > 1,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 12, fill: C.slate } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 13, bold: true } },
    xAxis: { min: 0, max: cfg.max ?? 1, majorUnit: cfg.unit ?? 0.2, majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 }, textStyle: { fontSize: 12, fill: C.slate } },
    yAxis: { textStyle: { fontSize: 13, fill: C.ink }, line: { style: "solid", fill: C.line, width: 1 } },
    barOptions: { direction: "bar", grouping: cfg.grouping ?? "clustered", gapWidth: 55 },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
  });
  addCard(slide, cfg.card, { left: 758, top: 184, width: 410, height: 420 }, cfg.accent2 ?? C.teal);
  notes(slide, cfg.sources, cfg.presenter);
}

function addAuditResult(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  addBox(slide, { left: 48, top: 184, width: 620, height: 390 }, { fill: C.white, line: C.line });
  slide.charts.add("bar", {
    position: { left: 82, top: 218, width: 550, height: 305 },
    categories: ["Standard", "Paired control", "Equivariant"],
    series: [{ name: "Worst layer swap residual", values: [0, 5.9, 0], fill: C.blue, points: [{ idx: 1, fill: C.red }, { idx: 2, fill: C.teal }] }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 14, bold: true } },
    xAxis: { min: 0, max: 6.5, majorUnit: 1, majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 }, textStyle: { fontSize: 12, fill: C.slate } },
    yAxis: { textStyle: { fontSize: 13, fill: C.ink } },
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 62 },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
  });
  addText(slide, "Lower is better; pass gate = 5×10⁻⁵", { left: 95, top: 531, width: 520, height: 24 }, { fontSize: 13, color: C.slate, alignment: "center" });
  addCard(slide, { kicker: "CHANNEL HEALTH", title: "Exact—and non-empty", body: "Odd/even energy ratio 0.022\nOdd effective rank 1.532\nAll registered health gates passed", accent: C.teal, fill: C.paleTeal, titleSize: 27, bodySize: 18 }, { left: 700, top: 184, width: 468, height: 188 }, C.teal);
  addCard(slide, { kicker: "BOUNDARY", title: "A geometry result", body: "96 sequences · 18 videos · 394 windows · 3 seeds\nNo force target. No participant-held-out clinical endpoint.", accent: C.orange, fill: C.paleOrange, titleSize: 27, bodySize: 17 }, { left: 700, top: 391, width: 468, height: 183 }, C.orange);
  notes(slide, cfg.sources, cfg.presenter);
}

function addFuture(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 58 }, { fontSize: 20, color: C.slate });
  const cols = [
    { x: 48, w: 190, title: "OBSERVE", body: "RGB video\nDepth\nWearables", fill: C.paleBlue, accent: C.blue },
    { x: 278, w: 215, title: "CANONICAL STATE", body: "Body · scene\nContact · dynamics\nUncertainty", fill: C.paleTeal, accent: C.teal },
    { x: 533, w: 215, title: "BIOMECH-JEPA", body: "Mask and predict\nFuture state\nCounterfactual state", fill: "#EEF2FF", accent: "#6D5BD0" },
    { x: 788, w: 190, title: "PHYSICS CHECK", body: "OpenSim / robot sim\nForce residuals\nGroundedness", fill: C.paleOrange, accent: C.orange },
    { x: 1018, w: 150, title: "ACT", body: "Estimate\nGenerate\nAsk tools", fill: C.paleRed, accent: C.red },
  ];
  cols.slice(0, -1).forEach((c, i) => addText(slide, "→", { left: c.x + c.w + 5, top: 298, width: 30, height: 40 }, { fontSize: 26, color: C.slate, bold: true, alignment: "center" }));
  cols.forEach((c) => {
    addBox(slide, { left: c.x, top: 214, width: c.w, height: 230 }, { fill: c.fill, line: c.accent });
    addText(slide, c.title, { left: c.x + 14, top: 239, width: c.w - 28, height: 30 }, { fontSize: 14, color: c.accent, bold: true, alignment: "center" });
    addText(slide, c.body, { left: c.x + 14, top: 294, width: c.w - 28, height: 110 }, { fontSize: c.w < 170 ? 16 : 18, color: C.navy, bold: true, alignment: "center", verticalAlignment: "middle" });
  });
  addBox(slide, { left: 130, top: 484, width: 1028, height: 108 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.bottomTitle, { left: 164, top: 506, width: 970, height: 28 }, { fontSize: 18, color: cfg.accent, bold: true, alignment: "center" });
  addText(slide, cfg.bottomBody, { left: 164, top: 542, width: 970, height: 34 }, { fontSize: 17, color: C.white, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addMeasurementGap(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 54 }, { fontSize: 20, color: C.slate });
  const nodes = [
    { x: 48, w: 230, kicker: "OBSERVED", title: "Pixels and pose", body: "Projection\nLandmark confidence\nOcclusion", fill: C.paleBlue, accent: C.blue },
    { x: 332, w: 230, kicker: "NORMALIZED", title: "Body-frame motion", body: "Joint trajectories\nAnatomical identity\nCycle phase", fill: C.paleTeal, accent: C.teal },
    { x: 660, w: 230, kicker: "LATENT PHYSICS", title: "Contact and dynamics", body: "Ground contact\nExternal force\nMass and control", fill: C.paleOrange, accent: C.orange },
    { x: 944, w: 224, kicker: "DECISION", title: "Meaningful output", body: "Asymmetry\nGroundedness\nUncertainty", fill: C.paleRed, accent: C.red },
  ];
  addText(slide, "→", { left: 286, top: 310, width: 36, height: 40 }, { fontSize: 28, color: C.slate, bold: true, alignment: "center" });
  addText(slide, "?", { left: 574, top: 288, width: 74, height: 70 }, { fontSize: 46, color: C.red, bold: true, alignment: "center" });
  addText(slide, "→", { left: 898, top: 310, width: 36, height: 40 }, { fontSize: 28, color: C.slate, bold: true, alignment: "center" });
  nodes.forEach((n) => {
    addBox(slide, { left: n.x, top: 208, width: n.w, height: 270 }, { fill: n.fill, line: n.accent });
    addText(slide, n.kicker, { left: n.x + 20, top: 232, width: n.w - 40, height: 22 }, { fontSize: 12, color: n.accent, bold: true, alignment: "center" });
    addText(slide, n.title, { left: n.x + 18, top: 277, width: n.w - 36, height: 52 }, { fontSize: 24, color: C.navy, bold: true, alignment: "center" });
    addText(slide, n.body, { left: n.x + 18, top: 355, width: n.w - 36, height: 90 }, { fontSize: 17, color: C.slate, alignment: "center" });
  });
  addText(slide, "The inverse problem", { left: 566, top: 373, width: 94, height: 46 }, { fontSize: 13, color: C.red, bold: true, alignment: "center" });
  addBox(slide, { left: 126, top: 516, width: 1028, height: 94 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.bottom, { left: 165, top: 541, width: 950, height: 48 }, { fontSize: 19, color: C.white, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addStakeRows(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const headers = ["SETTING", "PHYSICAL QUESTION", "FAILURE IF SIDE OR PHYSICS IS WRONG"];
  const xs = [48, 284, 720];
  const ws = [220, 420, 448];
  headers.forEach((h, i) => addText(slide, h, { left: xs[i] + 12, top: 184, width: ws[i] - 24, height: 24 }, { fontSize: 12, color: cfg.accent, bold: true }));
  cfg.rows.forEach((row, i) => {
    const top = 222 + i * 112;
    addBox(slide, { left: 48, top, width: 1120, height: 94 }, { fill: i % 2 === 0 ? C.panel : C.white, line: C.line });
    addText(slide, row.setting, { left: 68, top: top + 23, width: 180, height: 52 }, { fontSize: 20, color: row.accent ?? cfg.accent, bold: true });
    addText(slide, row.question, { left: 302, top: top + 18, width: 386, height: 62 }, { fontSize: 18, color: C.navy, bold: true });
    addText(slide, row.failure, { left: 738, top: top + 18, width: 408, height: 62 }, { fontSize: 17, color: C.slate });
  });
  addBox(slide, { left: 48, top: 580, width: 1120, height: 56 }, { fill: cfg.bottomFill ?? C.paleBlue, line: cfg.accent });
  addText(slide, cfg.bottom, { left: 76, top: 596, width: 1064, height: 28 }, { fontSize: 18, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addExperimentDesign(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const facts = [
    { x: 48, value: "96", label: "sequences" },
    { x: 232, value: "18", label: "source videos" },
    { x: 416, value: "394", label: "windows" },
    { x: 600, value: "3", label: "paired seeds" },
  ];
  facts.forEach((f) => {
    addText(slide, f.value, { left: f.x, top: 192, width: 150, height: 54 }, { fontSize: 34, color: cfg.accent, bold: true, alignment: "center" });
    addText(slide, f.label, { left: f.x, top: 246, width: 150, height: 26 }, { fontSize: 14, color: C.slate, alignment: "center" });
  });
  addBox(slide, { left: 786, top: 185, width: 382, height: 94 }, { fill: C.paleOrange, line: C.orange });
  addText(slide, "Seeds 7 · 19 · 31", { left: 810, top: 207, width: 334, height: 28 }, { fontSize: 22, color: C.navy, bold: true, alignment: "center" });
  addText(slide, "Compared run-for-run—not best against best", { left: 810, top: 242, width: 334, height: 20 }, { fontSize: 14, color: C.slate, alignment: "center" });
  addCard(slide, { kicker: "EXPOSURE-MATCHED", title: "Same data budget", body: "300 updates per model\n2,400 orbit exposures\nOriginal and reflected views matched", accent: C.blue, fill: C.paleBlue, titleSize: 25, bodySize: 17 }, { left: 48, top: 315, width: 350, height: 230 }, C.blue);
  addCard(slide, { kicker: "COMPUTE-MATCHED", title: "Same compute proxy", body: "≈183.2–183.7B\nStandard 393 updates\nPaired 155 · Equivariant 300", accent: C.teal, fill: C.paleTeal, titleSize: 25, bodySize: 17 }, { left: 433, top: 315, width: 350, height: 230 }, C.teal);
  addCard(slide, { kicker: "CAPACITY ACCOUNTING", title: "Report—not pretend—matching", body: "Standard 41,696 params\nPaired 84,256\nEquivariant 50,272", accent: C.orange, fill: C.paleOrange, titleSize: 24, bodySize: 17 }, { left: 818, top: 315, width: 350, height: 230 }, C.orange);
  addText(slide, cfg.bottom, { left: 86, top: 578, width: 1048, height: 38 }, { fontSize: 17, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addFutureDetailed(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 19, color: C.slate });
  addText(slide, "↓", { left: 592, top: 214, width: 38, height: 36 }, { fontSize: 28, color: C.slate, bold: true, alignment: "center" });
  addText(slide, "↓", { left: 592, top: 369, width: 38, height: 36 }, { fontSize: 28, color: C.slate, bold: true, alignment: "center" });
  const tools = ["RGB", "depth", "segmentation", "monocular motion", "wearables"];
  tools.forEach((t, i) => {
    const x = 48 + i * 226;
    addBox(slide, { left: x, top: 182, width: 196, height: 58 }, { fill: C.paleBlue, line: C.blue });
    addText(slide, t, { left: x + 10, top: 199, width: 176, height: 26 }, { fontSize: 16, color: C.navy, bold: true, alignment: "center" });
  });
  addBox(slide, { left: 160, top: 263, width: 960, height: 106 }, { fill: C.paleTeal, line: C.teal });
  addText(slide, "TYPED WORLD STATE", { left: 190, top: 281, width: 900, height: 24 }, { fontSize: 13, color: C.teal, bold: true, alignment: "center" });
  addText(slide, "body · scene · object keypoints · contact · dynamics · uncertainty", { left: 190, top: 319, width: 900, height: 34 }, { fontSize: 24, color: C.navy, bold: true, alignment: "center" });
  addBox(slide, { left: 160, top: 400, width: 960, height: 78 }, { fill: "#EEF2FF", line: "#6D5BD0" });
  addText(slide, "BIOMECH-JEPA predicts masked state, future state, and counterfactual state", { left: 190, top: 423, width: 900, height: 32 }, { fontSize: 21, color: C.navy, bold: true, alignment: "center" });
  const heads = [
    { x: 48, title: "DISCRIMINATIVE", body: "forces · asymmetry\ngroundedness · confidence", fill: C.paleTeal, accent: C.teal },
    { x: 430, title: "GENERATIVE", body: "SMPL / body keypoints\nobject trajectories", fill: C.paleBlue, accent: C.blue },
    { x: 812, title: "PHYSICS + AGENT", body: "OpenSim / robot simulation\nQwen tool policy with RL", fill: C.paleOrange, accent: C.orange },
  ];
  heads.forEach((h) => {
    addBox(slide, { left: h.x, top: 512, width: 356, height: 112 }, { fill: h.fill, line: h.accent });
    addText(slide, h.title, { left: h.x + 18, top: 530, width: 320, height: 20 }, { fontSize: 12, color: h.accent, bold: true, alignment: "center" });
    addText(slide, h.body, { left: h.x + 18, top: 561, width: 320, height: 48 }, { fontSize: 17, color: C.navy, bold: true, alignment: "center" });
  });
  notes(slide, cfg.sources, cfg.presenter);
}

function addCollaboration(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 48 }, { fontSize: 20, color: C.slate });
  const streams = [
    { x: 48, title: "BIOMECHANICS", subtitle: "Delp lab collaboration", body: "Force-plate endpoint\nContact and ground reaction\nOpenSim teacher / adjudicator\nParticipant protocol", fill: C.paleTeal, accent: C.teal },
    { x: 433, title: "AMBIENT INTELLIGENCE", subtitle: "Stanford HAI collaboration", body: "Real camera variation\nDepth and segmentation\nScene and object context\nUncertainty-aware sensing", fill: C.paleBlue, accent: C.blue },
    { x: 818, title: "WORLD-MODEL CORE", subtitle: "Shared research program", body: "Typed latent state\nJEPA objectives and generation\nPhysics residual learning\nActive tool routing with RL", fill: "#EEF2FF", accent: "#6D5BD0" },
  ];
  streams.forEach((s) => {
    addBox(slide, { left: s.x, top: 194, width: 350, height: 320 }, { fill: s.fill, line: s.accent });
    addText(slide, s.title, { left: s.x + 24, top: 220, width: 302, height: 24 }, { fontSize: 13, color: s.accent, bold: true, alignment: "center" });
    addText(slide, s.subtitle, { left: s.x + 24, top: 262, width: 302, height: 56 }, { fontSize: 24, color: C.navy, bold: true, alignment: "center" });
    addText(slide, s.body, { left: s.x + 36, top: 345, width: 278, height: 130 }, { fontSize: 17, color: C.slate, alignment: "center" });
  });
  addBox(slide, { left: 78, top: 552, width: 1060, height: 72 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.bottom, { left: 110, top: 573, width: 996, height: 32 }, { fontSize: 17, color: C.white, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addConclusion(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.statement, { left: 92, top: 145, width: 1096, height: 118 }, { fontSize: 34, color: C.navy, bold: true, alignment: "center" });
  const w = 330;
  cfg.points.forEach((p, i) => addCard(slide, p, { left: 90 + i * 365, top: 318, width: w, height: 220 }, cfg.accent));
  addBox(slide, { left: 128, top: 578, width: 1024, height: 60 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.bottom, { left: 160, top: 595, width: 960, height: 30 }, { fontSize: 18, color: C.white, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

async function addPhotoBullets(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  const imageLeft = cfg.imageSide !== "right";
  const imagePos = { left: imageLeft ? 48 : 714, top: 174, width: 518, height: 412 };
  const textLeft = imageLeft ? 604 : 48;
  addBox(slide, imagePos, { fill: C.panel, line: C.line });
  await addPhoto(slide, cfg.image, { left: imagePos.left + 8, top: imagePos.top + 8, width: imagePos.width - 16, height: imagePos.height - 16 }, cfg.alt);
  if (cfg.kicker) addText(slide, cfg.kicker.toUpperCase(), { left: textLeft, top: 184, width: 520, height: 24 }, { fontSize: 13, color: cfg.accent, bold: true });
  addText(slide, cfg.bodyTitle, { left: textLeft, top: cfg.kicker ? 220 : 190, width: 520, height: 72 }, { fontSize: 30, color: C.navy, bold: true });
  addBullets(slide, cfg.bullets, { left: textLeft, top: cfg.kicker ? 305 : 282, width: 520, height: 190 }, { fontSize: 22, color: C.ink });
  addBox(slide, { left: textLeft, top: 500, width: 520, height: 86 }, { fill: cfg.takeawayFill ?? C.paleBlue, line: cfg.accent });
  addText(slide, cfg.takeaway, { left: textLeft + 24, top: 521, width: 472, height: 48 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center", verticalAlignment: "middle" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addSimpleRows(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  const start = cfg.lead ? 182 : 145;
  const rowH = cfg.takeaway ? 118 : 140;
  cfg.rows.forEach((row, i) => {
    const top = start + i * rowH;
    addText(slide, row.marker ?? String(i + 1).padStart(2, "0"), { left: 62, top: top + 22, width: 90, height: 50 }, { fontSize: row.markerSize ?? 30, color: row.accent ?? cfg.accent, bold: true, alignment: "center" });
    addText(slide, row.title, { left: 176, top: top + 18, width: 350, height: 38 }, { fontSize: 25, color: C.navy, bold: true });
    addText(slide, `• ${row.body}`, { left: 548, top: top + 18, width: 570, height: 66 }, { fontSize: 20, color: C.slate });
    if (i < cfg.rows.length - 1) addRule(slide, 176, top + rowH - 14, 942, C.line, 1);
  });
  if (cfg.takeaway) {
    addBox(slide, { left: 176, top: 574, width: 942, height: 62 }, { fill: cfg.takeawayFill ?? C.paleBlue, line: cfg.accent });
    addText(slide, cfg.takeaway, { left: 202, top: 591, width: 890, height: 30 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

function addSimpleSteps(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  const top = 220;
  const n = cfg.steps.length;
  const gap = 34;
  const w = (1120 - gap * (n - 1)) / n;
  cfg.steps.forEach((step, i) => {
    const left = 48 + i * (w + gap);
    if (i < n - 1) addText(slide, "→", { left: left + w + 4, top: top + 65, width: 26, height: 38 }, { fontSize: 27, color: C.slate, bold: true, alignment: "center" });
    addText(slide, String(i + 1), { left: left + 6, top, width: 54, height: 54 }, { fontSize: 29, color: step.accent ?? cfg.accent, bold: true, alignment: "center", verticalAlignment: "middle" });
    addRule(slide, left + 8, top + 68, w - 16, step.accent ?? cfg.accent, 5);
    addText(slide, step.title, { left: left + 8, top: top + 92, width: w - 16, height: 66 }, { fontSize: 24, color: C.navy, bold: true, alignment: "center" });
    addText(slide, `• ${step.body}`, { left: left + 14, top: top + 172, width: w - 28, height: 98 }, { fontSize: 18, color: C.slate, alignment: "center" });
  });
  if (cfg.takeaway) {
    addBox(slide, { left: 148, top: 535, width: 984, height: 72 }, { fill: cfg.takeawayFill ?? C.paleBlue, line: cfg.accent });
    addText(slide, cfg.takeaway, { left: 180, top: 556, width: 920, height: 34 }, { fontSize: 20, color: C.navy, bold: true, alignment: "center" });
  }
  notes(slide, cfg.sources, cfg.presenter);
}

function addMetricBullets(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  const w = 330;
  cfg.metrics.forEach((m, i) => {
    const left = 48 + i * 380;
    addText(slide, m.value, { left, top: 196, width: w, height: 62 }, { fontSize: 43, color: m.accent, bold: true, alignment: "center" });
    addText(slide, m.label, { left, top: 263, width: w, height: 34 }, { fontSize: 21, color: C.navy, bold: true, alignment: "center" });
    addText(slide, m.note, { left: left + 24, top: 310, width: w - 48, height: 66 }, { fontSize: 17, color: C.slate, alignment: "center" });
  });
  addRule(slide, 82, 401, 1036, C.line, 1);
  addText(slide, cfg.bottomTitle, { left: 72, top: 435, width: 260, height: 36 }, { fontSize: 24, color: cfg.accent, bold: true });
  addBullets(slide, cfg.bullets, { left: 330, top: 432, width: 800, height: 150 }, { fontSize: 21, color: C.ink });
  notes(slide, cfg.sources, cfg.presenter);
}

function addTwoColumnBullets(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  if (cfg.lead) addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  addRule(slide, 608, 184, 1, C.line, 360);
  [cfg.left, cfg.right].forEach((col, i) => {
    const left = i === 0 ? 56 : 662;
    addText(slide, col.kicker.toUpperCase(), { left, top: 194, width: 500, height: 24 }, { fontSize: 13, color: col.accent, bold: true });
    addText(slide, col.title, { left, top: 232, width: 500, height: 60 }, { fontSize: 30, color: C.navy, bold: true });
    addBullets(slide, col.bullets, { left, top: 318, width: 500, height: 200 }, { fontSize: 21, color: C.ink });
  });
  addBox(slide, { left: 126, top: 570, width: 1028, height: 66 }, { fill: cfg.takeawayFill ?? C.paleBlue, line: cfg.accent });
  addText(slide, cfg.takeaway, { left: 158, top: 588, width: 964, height: 32 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addMirrorCompare(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  addText(slide, "L  ↔  R", { left: 76, top: 202, width: 470, height: 90 }, { fontSize: 55, color: C.red, bold: true, alignment: "center" });
  addText(slide, "Mirror the body", { left: 76, top: 300, width: 470, height: 46 }, { fontSize: 29, color: C.navy, bold: true, alignment: "center" });
  addBullets(slide, ["Left and right exchange", "Signed difference flips"], { left: 112, top: 371, width: 400, height: 112 }, { fontSize: 22 });
  addRule(slide, 630, 194, 1, C.line, 300);
  addText(slide, "same person     camera moves →", { left: 690, top: 213, width: 460, height: 72 }, { fontSize: 31, color: C.blue, bold: true, alignment: "center" });
  addText(slide, "Move the camera", { left: 690, top: 300, width: 460, height: 46 }, { fontSize: 29, color: C.navy, bold: true, alignment: "center" });
  addBullets(slide, ["Anatomy stays fixed", "Physical answer stays stable"], { left: 726, top: 371, width: 400, height: 112 }, { fontSize: 22 });
  addBox(slide, { left: 180, top: 541, width: 920, height: 70 }, { fill: C.paleBlue, line: cfg.accent });
  addText(slide, cfg.takeaway, { left: 215, top: 561, width: 850, height: 34 }, { fontSize: 20, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addNullResult(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  addText(slide, "Expected", { left: 72, top: 190, width: 260, height: 28 }, { fontSize: 16, color: C.teal, bold: true, alignment: "center" });
  addText(slide, "−1.000", { left: 72, top: 226, width: 260, height: 72 }, { fontSize: 46, color: C.teal, bold: true, alignment: "center" });
  addText(slide, "mirror slope", { left: 72, top: 300, width: 260, height: 28 }, { fontSize: 16, color: C.slate, alignment: "center" });
  addText(slide, "→", { left: 346, top: 240, width: 90, height: 60 }, { fontSize: 42, color: C.slate, bold: true, alignment: "center" });
  addText(slide, "Learned", { left: 450, top: 190, width: 260, height: 28 }, { fontSize: 16, color: C.red, bold: true, alignment: "center" });
  addText(slide, "−0.337", { left: 450, top: 226, width: 260, height: 72 }, { fontSize: 46, color: C.red, bold: true, alignment: "center" });
  addText(slide, "mirror slope", { left: 450, top: 300, width: 260, height: 28 }, { fontSize: 16, color: C.slate, alignment: "center" });
  addRule(slide, 745, 183, 1, C.line, 335);
  addBullets(slide, cfg.bullets, { left: 790, top: 206, width: 380, height: 250 }, { fontSize: 21, color: C.ink });
  addBox(slide, { left: 78, top: 385, width: 610, height: 118 }, { fill: C.paleRed, line: C.red });
  addText(slide, "R² = −0.0678", { left: 104, top: 405, width: 220, height: 44 }, { fontSize: 30, color: C.red, bold: true, alignment: "center" });
  addText(slide, "The embedding predicts worse than a constant baseline.", { left: 326, top: 411, width: 330, height: 62 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center" });
  addBox(slide, { left: 180, top: 552, width: 920, height: 62 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.takeaway, { left: 212, top: 569, width: 856, height: 30 }, { fontSize: 19, color: C.white, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addCollapseGate(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  addText(slide, "0 = −0", { left: 70, top: 218, width: 450, height: 110 }, { fontSize: 68, color: C.orange, bold: true, alignment: "center" });
  addText(slide, "Perfect symmetry. No information.", { left: 70, top: 345, width: 450, height: 48 }, { fontSize: 25, color: C.navy, bold: true, alignment: "center" });
  addRule(slide, 592, 190, 1, C.line, 330);
  addText(slide, "Before using the representation, check:", { left: 646, top: 212, width: 500, height: 54 }, { fontSize: 27, color: C.navy, bold: true });
  addBullets(slide, cfg.bullets, { left: 646, top: 304, width: 500, height: 180 }, { fontSize: 22 });
  addBox(slide, { left: 176, top: 552, width: 928, height: 64 }, { fill: C.paleOrange, line: C.orange });
  addText(slide, cfg.takeaway, { left: 206, top: 570, width: 868, height: 30 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addExperimentOverview(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  cfg.facts.forEach((fact, i) => {
    const left = 56 + i * 278;
    addText(slide, fact.value, { left, top: 185, width: 220, height: 56 }, { fontSize: 39, color: fact.accent ?? cfg.accent, bold: true, alignment: "center" });
    addText(slide, fact.label, { left, top: 245, width: 220, height: 28 }, { fontSize: 16, color: C.slate, alignment: "center" });
  });
  addRule(slide, 80, 300, 1040, C.line, 1);
  addText(slide, "Fair comparison", { left: 86, top: 337, width: 300, height: 44 }, { fontSize: 28, color: C.navy, bold: true });
  addBullets(slide, cfg.bullets, { left: 400, top: 332, width: 720, height: 170 }, { fontSize: 21 });
  addBox(slide, { left: 170, top: 548, width: 940, height: 66 }, { fill: C.paleOrange, line: C.orange });
  addText(slide, cfg.takeaway, { left: 202, top: 566, width: 876, height: 32 }, { fontSize: 19, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addAuditIntuitive(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  addBox(slide, { left: 48, top: 184, width: 690, height: 395 }, { fill: C.white, line: C.line });
  slide.charts.add("bar", {
    position: { left: 82, top: 224, width: 610, height: 280 },
    categories: ["Paired control", "Equivariant"],
    series: [{ name: "Mirror-rule error", values: [5.9, 0], fill: C.red, points: [{ idx: 1, fill: C.teal }] }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 16, bold: true } },
    xAxis: { min: 0, max: 6.5, majorUnit: 1, majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 }, textStyle: { fontSize: 13, fill: C.slate } },
    yAxis: { textStyle: { fontSize: 16, fill: C.ink } },
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 75 },
    chartFill: C.white,
    chartLine: { style: "solid", width: 0, fill: C.white },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", width: 0, fill: C.white },
  });
  addText(slide, "Mirror-rule error: lower is better", { left: 110, top: 524, width: 560, height: 28 }, { fontSize: 15, color: C.slate, alignment: "center" });
  addText(slide, "What this means", { left: 790, top: 196, width: 360, height: 42 }, { fontSize: 29, color: C.navy, bold: true });
  addBullets(slide, cfg.bullets, { left: 790, top: 270, width: 360, height: 230 }, { fontSize: 21 });
  addBox(slide, { left: 790, top: 500, width: 360, height: 80 }, { fill: C.paleOrange, line: C.orange });
  addText(slide, cfg.takeaway, { left: 812, top: 519, width: 316, height: 44 }, { fontSize: 18, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addFutureIntuitive(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.lead, { left: 48, top: 118, width: 1120, height: 44 }, { fontSize: 20, color: C.slate });
  const stages = [
    { title: "Observe", body: "• RGB video\n• depth\n• segmentation", accent: C.blue },
    { title: "Build state", body: "• body + objects\n• contact\n• uncertainty", accent: C.teal },
    { title: "Check physics", body: "• OpenSim\n• robot simulation\n• groundedness", accent: C.orange },
    { title: "Answer or act", body: "• estimate\n• generate\n• call a tool", accent: C.red },
  ];
  stages.forEach((stage, i) => {
    const left = 48 + i * 286;
    if (i < 3) addText(slide, "→", { left: left + 258, top: 316, width: 28, height: 38 }, { fontSize: 27, color: C.slate, bold: true, alignment: "center" });
    addText(slide, stage.title, { left, top: 208, width: 246, height: 50 }, { fontSize: 27, color: C.navy, bold: true, alignment: "center" });
    addRule(slide, left + 22, 274, 202, stage.accent, 5);
    addText(slide, stage.body, { left: left + 26, top: 305, width: 194, height: 150 }, { fontSize: 19, color: C.ink, alignment: "left" });
  });
  addBox(slide, { left: 135, top: 510, width: 1010, height: 96 }, { fill: "#EEF2FF", line: cfg.accent });
  addText(slide, cfg.takeaway, { left: 170, top: 535, width: 940, height: 50 }, { fontSize: 20, color: C.navy, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

function addConclusionIntuitive(presentation, cfg, page) {
  const slide = addBase(presentation, cfg.title, cfg.section, page, cfg.accent);
  addText(slide, cfg.statement, { left: 86, top: 140, width: 1108, height: 86 }, { fontSize: 32, color: C.navy, bold: true, alignment: "center" });
  addBullets(slide, cfg.bullets, { left: 230, top: 270, width: 820, height: 190 }, { fontSize: 25, color: C.ink });
  addBox(slide, { left: 145, top: 512, width: 990, height: 96 }, { fill: C.navy, line: C.navy });
  addText(slide, cfg.bottom, { left: 185, top: 538, width: 910, height: 48 }, { fontSize: 21, color: C.white, bold: true, alignment: "center" });
  notes(slide, cfg.sources, cfg.presenter);
}

async function buildHighSchool() {
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const accent = C.blue;
  addCover(p, {
    accent,
    eyebrow: "Advanced high-school research talk",
    title: "GaitParity: Teaching AI Which Side of the Body Is Which",
    subtitle: "A mirror test for movement models—and a path from video skeletons toward trustworthy biomechanics.",
    bottom: "Question → physical rule → controlled audit → force-plate study → physics-grounded world model",
    sources: [SRC.pdf, SRC.current],
    presenter: "Open with the scientific question: can a movement model keep left and right straight when the body is mirrored?",
  });
  addTwoColumn(p, {
    title: "Why left and right are not interchangeable",
    section: "Motivation",
    accent,
    lead: "The same movement can look similar while each leg contributes very differently.",
    left: { kicker: "BIOMECHANICS", title: "Balance is often a side-to-side problem", body: "After stroke, one leg may push less or support the body differently.\n\nForce plates measure that difference. Video alone can hide it.", accent: C.teal, fill: C.paleTeal },
    right: { kicker: "AMBIENT INTELLIGENCE", title: "Cameras can create a false side change", body: "A flipped feed, a new viewpoint, or a missing heel can make the same person look different.\n\nA useful room-scale system must separate anatomy from observation.", accent: C.orange, fill: C.paleOrange },
    takeaway: "A body mirror should flip a signed asymmetry. A camera move should not.",
    sources: [SRC.pdf, SRC.method],
  }, 2);
  await addImageSlide(p, {
    title: "The earlier project built a complete motion pipeline",
    section: "Previous work",
    accent,
    lead: "Public gait video became a fixed-length skeleton sequence, then a self-supervised motion representation.",
    image: "prior-sjepa-pipeline.png",
    alt: "Prior S-JEPA pipeline from public video to audited readouts",
    callout: "96 canonical sequences came from 18 source videos. The source video—not the clip—is the real independence boundary.",
    sources: [SRC.nb01, path.join(IMG, "prior-sjepa-pipeline.svg")],
  }, 3);
  addPipeline(p, {
    title: "S-JEPA learns by predicting what was hidden",
    section: "Method",
    accent,
    lead: "The model does not reconstruct pixels. It predicts a useful representation of missing movement.",
    steps: [
      { title: "Observe", body: "64 frames\n33 body landmarks\nvisibility carried along", fill: C.paleBlue, accent: C.blue },
      { title: "Hide", body: "Mask selected joints and times without inventing a fake injury", fill: C.paleOrange, accent: C.orange },
      { title: "Predict", body: "Use visible motion to predict the hidden target features", fill: "#EEF2FF", accent: "#6D5BD0" },
      { title: "Audit", body: "Check feature variation, confounds, splits, and downstream readouts", fill: C.paleTeal, accent: C.teal },
    ],
    takeaway: "The objective teaches motion structure. It does not automatically teach physics or left–right meaning.",
    sources: [SRC.nb04, SRC.method],
  }, 4);
  addMetricSlide(p, {
    title: "The training run learned—but did not become a diagnosis",
    section: "Previous results",
    accent,
    lead: "The representation stayed varied and the prediction loss fell substantially.",
    metrics: [
      { value: "6.56 → 0.59", label: "JEPA loss", note: "Normal-only Stage 0, 300 epochs", accent: C.blue, fill: C.paleBlue },
      { value: ".296 → .465", label: "feature spread", note: "Evidence against total constant collapse", accent: C.teal, fill: C.paleTeal },
      { value: "0.414", label: "final feature std", note: "After five condition-aware stages", accent: C.orange, fill: C.paleOrange },
    ],
    bottomTitle: "What this supports",
    bottomBody: "The learning system ran correctly and produced nontrivial features. These are training-health results—not clinical validation.",
    sources: [SRC.nb04],
  }, 5);
  addBarResult(p, {
    title: "Condition labels were readable inside the exposed corpus",
    section: "Previous results",
    accent,
    lead: "The harder question is whether the same signal survives new source videos and new people.",
    categories: ["Missingness only", "S-JEPA features", "Grouped N vs A"],
    series: [{ name: "Accuracy", values: [0.414, 0.724, 0.843], fill: C.blue, points: [{ idx: 0, fill: C.slate }, { idx: 2, fill: C.teal }] }],
    max: 1,
    unit: 0.2,
    card: { kicker: "READ CAREFULLY", title: "Useful signal, limited claim", body: "All-96 five-class accuracy: 0.724\nGrouped normal-vs-abnormal: 0.843\n\nThe encoder had still seen every held-out classifier row. This is not unseen-person performance.", accent: C.orange, fill: C.paleOrange, titleSize: 28, bodySize: 17 },
    sources: [SRC.nb06],
  }, 6);
  await addImageSlide(p, {
    title: "The anatomical mirror creates a testable rule",
    section: "GaitParity",
    accent,
    lead: "Mirroring the body exchanges anatomy. Moving the camera only changes how the same anatomy is observed.",
    image: "mirror-versus-camera.png",
    alt: "Comparison of anatomical mirror and camera movement",
    callout: "Overall pace should stay similar. A right-minus-left quantity must reverse sign.",
    sources: [SRC.method, path.join(IMG, "mirror-versus-camera.svg")],
  }, 7);
  await addImageSlide(p, {
    title: "The first signed-laterality test failed",
    section: "GaitParity result 1",
    accent: C.red,
    lead: "A simple reader could not recover the constructed left–right score from the old representation.",
    image: "notebook-05a-signed-laterality-probe.png",
    alt: "Signed laterality probe scatter plots",
    callout: "Learned R² = −0.068 · mirror slope = −0.337, not near −1 · sign consistency = 0.444",
    calloutFill: C.paleRed,
    calloutLine: C.red,
    sources: [SRC.late, path.join(IMG, "notebook-05a-signed-laterality-probe.png")],
    presenter: "Call this an informative null result: it stops the project from presenting a weak side signal as a discovery.",
  }, 8);
  await addImageSlide(p, {
    title: "The next experiment compares three matched models",
    section: "GaitParity method",
    accent,
    lead: "The hard baseline is not an ordinary model. Three lines of arithmetic can already force the final answer to flip.",
    image: "model-comparisons.png",
    alt: "Standard, paired-unconstrained, and reflection-equivariant model comparison",
    callout: "The equivariant model must beat output repair and equally capable two-branch fusion—not merely a weaker one-view model.",
    sources: [SRC.long, path.join(IMG, "model-comparisons.svg")],
  }, 9);
  addAuditResult(p, {
    title: "The controlled audit can detect broken mirror geometry",
    section: "GaitParity result 2",
    accent: C.teal,
    lead: "Across matched runs, the equivariant encoder followed the rule exactly; the unrestricted control did not.",
    sources: [SRC.parity, path.join(IMG, "notebook-09f-full-gavd-audit.png")],
  }, 10);
  addTwoColumn(p, {
    title: "What the current evidence supports—and where it stops",
    section: "Discussion",
    accent: C.teal,
    lead: "Passing a rule is not the same as solving a biomechanical task.",
    left: { kicker: "SUPPORTED NOW", title: "A trustworthy research instrument", body: "✓ Correct anatomical reflection\n✓ Exact layerwise swap behavior\n✓ Audit catches the unrestricted control\n✓ Odd channel passes registered health gates", accent: C.teal, fill: C.paleTeal },
    right: { kicker: "NOT YET SUPPORTED", title: "A useful clinical measurement", body: "No force-plate target\nNo participant-held-out endpoint\nNo balance or diagnosis result\nNo GPU replication result", accent: C.red, fill: C.paleRed },
    takeaway: "The honest claim is engineering feasibility plus a clear reason to run the force-plate study.",
    sources: [SRC.parity, SRC.current],
  }, 11);
  await addImageSlide(p, {
    title: "The decisive study gives one person one vote",
    section: "Planned study",
    accent: C.orange,
    lead: "All cycles from one participant stay together during training, tuning, and testing.",
    image: "participant-safe-split.png",
    alt: "Participant-safe outer split for force-plate evaluation",
    callout: "Primary question: does the equivariant encoder reduce participant-level force MAE beyond output repair and the paired control?",
    calloutFill: C.paleOrange,
    calloutLine: C.orange,
    sources: [SRC.method, SRC.planned, path.join(IMG, "participant-safe-split.svg")],
  }, 12);
  addFuture(p, {
    title: "Future direction: a physics-grounded movement world model",
    section: "Future work",
    accent: C.blue,
    lead: "GaitParity can become one structural rule inside a larger Biomech-JEPA that represents the body, scene, contact, dynamics, and uncertainty.",
    bottomTitle: "The first decisive experiment should start from clean biomechanical state—not raw video.",
    bottomBody: "Then bridge to ambient video with depth, segmentation, monocular motion, simulation teachers, and an active Qwen tool agent.",
    sources: [SRC.mentor, SRC.long, SRC.planned],
  }, 13);
  addConclusion(p, {
    title: "The mirror rule turns a vague goal into a falsifiable study",
    section: "Conclusion",
    accent,
    statement: "We are not yet estimating force from everyday video. We now have a precise rule, a model that follows it, and an experiment that can decide whether it matters.",
    points: [
      { kicker: "LEARNED", title: "Motion structure", body: "Earlier S-JEPA runs produced nontrivial but confounded representations.", accent: C.blue, fill: C.paleBlue, titleSize: 25 },
      { kicker: "FIXED", title: "Left–right geometry", body: "The equivariant encoder passes the exact mirror contract across matched audits.", accent: C.teal, fill: C.paleTeal, titleSize: 25 },
      { kicker: "NEXT", title: "Physical meaning", body: "Force plates and new participants must test whether structure becomes useful.", accent: C.orange, fill: C.paleOrange, titleSize: 25 },
    ],
    bottom: "A stronger claim requires a stronger gate.",
    sources: [SRC.current, SRC.planned, SRC.mentor],
  }, 14);
  return p;
}

async function buildGeneral() {
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const accent = C.teal;
  addCover(p, {
    accent,
    eyebrow: "General audience research briefing",
    title: "GaitParity: From Movement Video to Trustworthy Left–Right Gait Measures",
    subtitle: "A reflection-aware world model for biomechanics and ambient intelligence—built with explicit tests for what it knows.",
    bottom: "Current result: exact mirror geometry. Next test: participant-held-out force-plate prediction.",
    sources: [SRC.pdf, SRC.current],
  });
  addThreeCards(p, {
    title: "The opportunity is measurement outside the lab",
    section: "Motivation",
    accent,
    subtitle: "Ambient cameras can observe many more steps than a short clinical visit—but observation is not mechanics.",
    cards: [
      { kicker: "SCALE", title: "Everyday movement", body: "Homes and clinics could observe how gait changes across days, fatigue, medication, or recovery.", accent: C.blue, fill: C.paleBlue },
      { kicker: "MEANING", title: "Which side contributes?", body: "Balance and propulsion depend on what each leg does—not only where landmarks appear in the image.", accent: C.teal, fill: C.paleTeal },
      { kicker: "TRUST", title: "Separate body from camera", body: "A flipped image, new viewpoint, or occluded heel must not become a false biomechanical change.", accent: C.orange, fill: C.paleOrange },
    ],
    takeaway: "The system needs physical movement understanding, not just pose tracking.",
    sources: [SRC.pdf, SRC.method],
  }, 2);
  addTwoColumn(p, {
    title: "A side error has two different consequences",
    section: "Motivation",
    accent,
    left: { kicker: "BIOMECHANICS / BALANCE", title: "Wrong sign, wrong interpretation", body: "A right-minus-left measure tells us which side contributes more. If the sign silently flips, the result can point to the wrong limb or hide compensation.", accent: C.teal, fill: C.paleTeal },
    right: { kicker: "AMBIENT INTELLIGENCE", title: "Wrong cause, wrong action", body: "An ambient system must know whether the person changed, the camera changed, or the pose estimator simply lost a landmark.", accent: C.orange, fill: C.paleOrange },
    takeaway: "GaitParity isolates the first requirement with an anatomical-mirror test.",
    sources: [SRC.method, SRC.long],
  }, 3);
  await addImageSlide(p, {
    title: "Previous work established the full learning pipeline",
    section: "Previous work",
    accent,
    lead: "Video → skeletons → masked motion prediction → fixed representations → audited readouts.",
    image: "foundations-pipeline.png",
    alt: "Foundation notebook pipeline",
    callout: "The pipeline is valuable as a research instrument; its data boundaries prevent clinical claims from the current corpus.",
    sources: [SRC.nb01, SRC.nb04, SRC.nb05, SRC.nb06, path.join(IMG, "foundations-pipeline.svg")],
  }, 4);
  addMetricSlide(p, {
    title: "The earlier model learned motion structure",
    section: "Previous results",
    accent,
    lead: "Training remained finite, feature variation stayed healthy, and condition information was readable.",
    metrics: [
      { value: "0.558", label: "final JEPA loss", note: "After the five-stage curriculum", accent: C.blue, fill: C.paleBlue },
      { value: "0.414", label: "final feature std", note: "No obvious total constant collapse", accent: C.teal, fill: C.paleTeal },
      { value: "0.724", label: "all-96 accuracy", note: "Descriptive; source-exposed and label-informed", accent: C.orange, fill: C.paleOrange },
    ],
    bottomTitle: "Interpretation",
    bottomBody: "The system found structure in the corpus. It did not establish performance on a new source video, new patient, or physical outcome.",
    sources: [SRC.nb04, SRC.nb06],
  }, 5);
  addTwoColumn(p, {
    title: "The main limitation was not model size—it was evidence design",
    section: "Previous results",
    accent: C.orange,
    lead: "Many clips can still come from the same source video, and the encoder had seen every classifier-test row.",
    left: { kicker: "WHAT LOOKED PROMISING", title: "Grouped normal-vs-abnormal", body: "Accuracy 0.843\nMacro-F1 0.821\nROC-AUC 0.952\n\nThe classifier split by source video.", accent: C.teal, fill: C.paleTeal },
    right: { kicker: "WHY CLAIMS STAY NARROW", title: "Encoder-transductive", body: "All 159 rows—including classifier-test rows—were used during representation training. The result is in-corpus signal, not unseen-source generalization.", accent: C.red, fill: C.paleRed },
    takeaway: "The new study moves the independence boundary to the participant before any fitting.",
    sources: [SRC.nb06, SRC.method],
  }, 6);
  await addImageSlide(p, {
    title: "The mirror rule separates anatomy from observation",
    section: "GaitParity",
    accent,
    lead: "A body mirror swaps left and right. A camera move observes the same left and right from somewhere else.",
    image: "gait-parity-core.png",
    alt: "Core GaitParity mirror-versus-camera concept",
    callout: "Invariant to camera change. Equivariant—predictably sign-flipping—under anatomical reflection.",
    sources: [SRC.method, path.join(IMG, "gait-parity-core.svg")],
  }, 7);
  await addImageSlide(p, {
    title: "The old representation failed the signed mirror test",
    section: "GaitParity result 1",
    accent: C.red,
    lead: "This was an informative null result, not a reason to abandon the question.",
    image: "notebook-05a-signed-laterality-probe.png",
    alt: "Signed laterality probe result",
    callout: "Learned R² = −0.068 · mirror slope = −0.337 · sign consistency = 0.444",
    calloutFill: C.paleRed,
    calloutLine: C.red,
    sources: [SRC.late, path.join(IMG, "notebook-05a-signed-laterality-probe.png")],
  }, 8);
  await addImageSlide(p, {
    title: "Three models isolate what the mirror-aware architecture adds",
    section: "Method",
    accent,
    lead: "Every serious comparison sees the original walk and its mirror and returns an exactly sign-flipping output.",
    image: "model-comparisons.png",
    alt: "Three matched GaitParity model variants",
    callout: "Only the equivariant model is forced to keep the mirror rule organized throughout its internal layers.",
    sources: [SRC.long, path.join(IMG, "model-comparisons.svg")],
  }, 9);
  addAuditResult(p, {
    title: "The new encoder passes the exact geometry audit",
    section: "GaitParity result 2",
    accent,
    lead: "The deliberately unrestricted model breaks the rule. The equivariant model satisfies it across layers, regimes, and seeds.",
    sources: [SRC.parity, path.join(IMG, "notebook-09f-full-gavd-audit.png")],
  }, 10);
  await addImageSlide(p, {
    title: "The evidence has advanced one rung—not the entire ladder",
    section: "Discussion",
    accent,
    lead: "The project now has a checked instrument and a successful local feasibility audit.",
    image: "current-evidence-status.png",
    alt: "Current evidence status from early probe through planned clinical test",
    callout: "The next meaningful claim requires force measurements and people the model has not encountered.",
    sources: [SRC.current, path.join(IMG, "current-evidence-status.svg")],
  }, 11);
  await addImageSlide(p, {
    title: "The planned study tests usefulness, not just correctness",
    section: "Planned study",
    accent: C.orange,
    lead: "Force plates provide an independent signed propulsion target. People—not gait cycles—define the outer split.",
    image: "planned-research-ladder.png",
    alt: "Research ladder from local audit to replication",
    callout: "Success requires a practically meaningful force-MAE advantage over output repair and matched two-branch fusion.",
    calloutFill: C.paleOrange,
    calloutLine: C.orange,
    sources: [SRC.planned, SRC.long, path.join(IMG, "planned-research-ladder.svg")],
  }, 12);
  addFuture(p, {
    title: "Beyond GaitParity: Biomech-JEPA",
    section: "Future work",
    accent,
    lead: "The larger research direction is a typed human-motion world model that can estimate, generate, and judge whether movement is physically grounded.",
    bottomTitle: "Simulation should teach and adjudicate; the learned model should amortize the pipeline.",
    bottomBody: "A Qwen agent can actively call depth, segmentation, monocular motion, and physics tools when uncertainty justifies the cost.",
    sources: [SRC.mentor, SRC.long, SRC.planned],
  }, 13);
  addConclusion(p, {
    title: "A trustworthy movement model must respect anatomy before claiming mechanics",
    section: "Conclusion",
    accent,
    statement: "GaitParity converts left–right understanding into a rule that can fail, a control that should fail, and a study that can decide whether the rule improves physical measurement.",
    points: [
      { kicker: "CURRENT", title: "Exact geometry", body: "The equivariant implementation passes the intended mirror contract.", accent: C.teal, fill: C.paleTeal },
      { kicker: "LIMIT", title: "No force claim", body: "Current GAVD evidence is local, transductive, and outcome-free.", accent: C.red, fill: C.paleRed },
      { kicker: "PROMISE", title: "A stronger world model", body: "Body, scene, contact, dynamics, and uncertainty can become one predictive state.", accent: C.blue, fill: C.paleBlue },
    ],
    bottom: "The right next step is the participant-held-out force-plate study.",
    sources: [SRC.current, SRC.planned, SRC.mentor],
  }, 14);
  return p;
}

async function buildTechnical() {
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const accent = "#6D5BD0";
  addCover(p, {
    accent,
    eyebrow: "Technical lab meeting",
    title: "GaitParity: Reflection-Equivariant S-JEPA for Signed Gait Asymmetry",
    subtitle: "A controlled test of whether movement representations preserve anatomical left and right.",
    bottom: "Current result: exact mirror geometry. Next decision: held-out force prediction.",
    sources: [SRC.nb04, SRC.late, SRC.parity, SRC.long],
    presenter: `Today I will describe GaitParity, a controlled study of whether a movement model preserves left and right anatomy when a body is mirrored. This is a small but important test within a larger goal: learning physical movement from ordinary video, rather than stopping at pose or joint trajectories.

I will begin with the gap between what a camera observes and what biomechanics needs. I will then summarize the earlier S-JEPA results, explain the signed-laterality failure that motivated this study, present the controlled parity experiment, and end with the force-plate study and Biomech-JEPA research direction.`,
  });
  await addPhotoBullets(p, {
    title: "Video shows movement—not the forces behind it",
    section: "Motivation",
    accent,
    lead: "Ambient cameras can observe walking in natural settings, but they do not directly measure the physical state.",
    image: "ucdavis_ambient_gait.jpg",
    alt: "Older adult walking in a hallway while wall-mounted cameras record gait",
    kicker: "What the camera misses",
    bodyTitle: "The same visible motion can have different causes",
    bullets: ["Video gives pixels and projected joints", "Force, contact, and muscle control stay hidden", "Occlusion can support several physical explanations"],
    takeaway: "Infer the hidden physics—and report when the evidence is weak.",
    sources: [SRC.method, SRC.planned, SRC.mentor, "https://health.ucdavis.edu/news/headlines/innovative-research-looks-at-walking-and-healthy-aging/2024/08", "UC Davis Health photograph: Patten_URC_exwidebody1.jpg"],
    presenter: `The central problem is that video gives us an observation of movement, not a direct measurement of the physical causes. Two people can produce similar joint trajectories while using different muscle strategies, distributing weight differently, or contacting the ground in different ways. Even for one person, partial occlusion can make several physical explanations consistent with the same image.

That means the model should not behave like a deterministic simulator that claims one certain answer from pixels. It should estimate a hidden physical state, express uncertainty when the video is ambiguous, and use additional tools or measurements when the observation is not sufficient.`,
  }, 2);
  await addPhotoBullets(p, {
    title: "Biomechanics asks what generated the motion",
    section: "Motivation",
    accent,
    lead: "A joint trajectory describes motion. A biomechanical measurement explains support, force, and control.",
    image: "wikimedia_force_plate_lab.jpg",
    imageSide: "right",
    alt: "Biomechanics laboratory with motion-capture cameras and floor-mounted force plates",
    kicker: "Why the physics matters",
    bodyTitle: "A pose can look right while the conclusion is wrong",
    bullets: ["Stroke: which leg contributes less propulsion?", "Balance: is the body still recoverably stable?", "Follow-up care: did the person change—or the camera?"],
    takeaway: "Signed left–right differences are a clear first test because mirroring should reverse them.",
    sources: [SRC.pdf, SRC.method, SRC.planned, "https://commons.wikimedia.org/wiki/File:Biomechanics_lab_with_force_plate.jpg", "Photo: Apfriesen, CC BY-SA 4.0"],
    presenter: `This gap matters because the clinical question is often about how movement is produced, not simply what the pose looks like. In stroke gait, for example, we may care about how much each leg contributes to forward propulsion. In balance assessment, we need to know where support and contact are changing, especially near a loss of stability.

We also need to separate a measured direction from a clinical interpretation. A negative right-minus-left propulsion value tells us which side produced less force in that measurement. It does not, by itself, identify the impaired side, because people may compensate in ways that redistribute force. Keeping those concepts separate prevents a mathematically correct sign from becoming an incorrect clinical claim.`,
  }, 3);
  addSimpleRows(p, {
    title: "Ambient settings change the observation—not always the person",
    section: "Motivation",
    accent,
    lead: "The model needs a different response for each kind of change.",
    rows: [
      { marker: "VIEW", markerSize: 18, title: "New camera angle", body: "Keep the anatomical answer stable after body-frame normalization.", accent: C.blue },
      { marker: "MIRROR", markerSize: 18, title: "Mirrored video", body: "Reverse a signed left–right measurement because anatomy is exchanged.", accent: C.red },
      { marker: "MASK", markerSize: 18, title: "Missing joint", body: "Increase uncertainty; do not flip the predicted side because a heel disappears.", accent: C.orange },
    ],
    takeaway: "One rule for every video augmentation is not enough.",
    sources: [SRC.method, SRC.long, SRC.mentor],
    presenter: `Ambient intelligence adds another layer of difficulty: the camera and the environment change even when the person does not. A new viewpoint changes projection and occlusion. A mirrored video exchanges anatomical left and right. A missing heel or knee should increase uncertainty, but it should not automatically reverse the estimated side.

These cases require different model responses. The representation should ignore camera-related nuisance changes, transform predictably when anatomy is mirrored, and remain sensitive to genuine changes in dynamics. One generic instruction to “make every augmented view look the same” cannot express all three requirements. We need selective symmetry, where the response depends on what the transformation means physically.`,
  }, 4);
  addMirrorCompare(p, {
    title: "A body mirror and a camera move mean different things",
    section: "Motivation",
    accent,
    lead: "A reflection changes anatomical identity. A viewpoint change only changes how anatomy is observed.",
    takeaway: "Right-minus-left propulsion flips under a body mirror, but not under a camera move.",
    sources: [SRC.method, SRC.long],
    presenter: `This slide makes the distinction concrete. If we reflect the body, left and right anatomy exchange roles. A signed quantity such as right-minus-left propulsion should therefore reverse sign. We call that an odd target. A quantity such as overall walking pace should stay the same under reflection, so it is an even target.

A camera move is different. It changes the image coordinates and may hide different joints, but it does not exchange the person’s anatomy. After we place the motion in a consistent body-centered coordinate system, both pace and anatomical propulsion difference should remain stable. This simple rule becomes the measurement contract for the rest of the study.`,
  }, 5);
  addSimpleSteps(p, {
    title: "Each experiment earns one stronger claim",
    section: "Research question",
    accent,
    lead: "Passing a lower step cannot substitute for the next test.",
    steps: [
      { title: "Rule", body: "Does mirroring produce the exact response?", accent: C.teal },
      { title: "Information", body: "Are the parity channels non-empty?", accent: C.teal },
      { title: "Utility", body: "Does parity improve held-out force prediction?", accent: C.orange },
      { title: "Replication", body: "Does it survive new views and a new cohort?", accent: C.blue },
    ],
    takeaway: "Current position: rule + information. Next gate: force utility.",
    sources: [SRC.long, SRC.planned],
    presenter: `We separate the evidence into four levels because each level supports a different claim. First, does the architecture obey the mirror rule exactly? Second, do the left-right channels contain nontrivial, participant-varying information rather than collapsing to zero? Third, does that organization improve prediction of a meaningful physical target on people who were not used for fitting? Finally, does the result survive new views and an independent replication cohort?

The current work reaches the first two levels locally. The decisive question is still ahead: does internal reflection structure improve force prediction beyond a simple sign-corrected output or a generic two-view model?`,
  }, 6);
  addSimpleSteps(p, {
    title: "The earlier S-JEPA pipeline made the test possible",
    section: "Foundation",
    accent,
    lead: "The model learns motion by predicting hidden landmarks from the visible context.",
    steps: [
      { title: "Video", body: "Public gait clips grouped by source video", accent: C.blue },
      { title: "Skeleton", body: "64-frame body-centered motion windows", accent: C.teal },
      { title: "Mask + predict", body: "33 context landmarks predict 12 eligible targets", accent: C.orange },
      { title: "Audit", body: "Readouts test information and transformation rules", accent: "#6D5BD0" },
    ],
    takeaway: "96 sequences come from only 18 videos—so source boundaries matter.",
    sources: [SRC.nb01, SRC.nb04],
    presenter: `The earlier S-JEPA work established the pipeline that makes this question testable. Public gait videos were converted into tracked skeleton sequences, normalized into fixed 64-frame windows, and divided into context landmarks and masked target landmarks. The model then learned to predict missing motion features without using a clinical label for every training step.

The key bookkeeping unit is the source video. The dataset contains 96 canonical sequences derived from only 18 uploaded recordings, so several clips can share the same person, camera, and environment. Treating those clips as independent would overstate generalization. Every later evaluation therefore needs to respect the source or participant boundary before any component is fitted.`,
  }, 7);
  addMetricBullets(p, {
    title: "The foundation learned motion structure—not a diagnosis",
    section: "Foundation evidence",
    accent,
    lead: "Three numbers separate training health from scientific meaning.",
    metrics: [
      { value: "0.558", label: "final prediction loss", note: "Training improved from 6.557 and stayed stable.", accent: C.blue },
      { value: "0.414", label: "feature variability", note: "The full representation did not collapse to one constant.", accent: C.teal },
      { value: "0.0278", label: "condition separation", note: "The five condition groups still overlap strongly.", accent: C.red },
    ],
    bottomTitle: "Interpretation",
    bullets: ["The optimization worked", "The features contain motion variation", "The latent space is not a clean clinical taxonomy"],
    sources: [SRC.nb04, SRC.nb05],
    presenter: `The foundation model trained successfully in the narrow engineering sense. Its prediction loss fell from 6.557 at the beginning of the curriculum to 0.558, and the final feature standard deviation of 0.414 argues against the entire representation becoming constant.

However, successful optimization is not the same as scientific understanding. The cosine silhouette score is only 0.0278, which means the five condition groups overlap strongly in the learned space. These results support a limited statement: the encoder learned nontrivial motion structure. They do not show that it recovered a physical variable, separated diagnoses cleanly, or generalized to a new patient.`,
  }, 8);
  addTwoColumnBullets(p, {
    title: "Strong readout scores did not prove prospective performance",
    section: "Foundation evidence",
    accent: C.orange,
    lead: "Condition information was readable, but the evaluation split happened too late.",
    left: { kicker: "WHAT WE SAW", title: "Readable condition signal", bullets: ["Accuracy: 0.843", "Macro-F1: 0.821", "ROC-AUC: 0.952"], accent: C.teal },
    right: { kicker: "WHY THE CLAIM IS LIMITED", title: "The encoder saw the test rows", bullets: ["All 159 rows entered representation training", "Later stages used condition labels", "Only the final readout was grouped"], accent: C.red },
    takeaway: "Split people or sources before every fitted stage—not only before the last classifier.",
    sources: [SRC.nb06, SRC.method],
    presenter: `The earlier classification results looked promising at first glance. A grouped normal-versus-abnormal readout reached 0.843 accuracy, 0.821 macro-F1, and 0.952 ROC-AUC. But the representation encoder had already trained on every row later used by the classifier test, and later curriculum stages also used condition labels.

So these numbers demonstrate that condition information is readable within this corpus; they do not establish prospective performance. A clean split at the final classifier is not enough if the encoder has already seen the evaluation examples. This lesson directly shapes GaitParity: sources or participants must be separated before every fitted stage, not only before the last readout.`,
  }, 9);
  addNullResult(p, {
    title: "The saved representation did not preserve signed laterality",
    section: "GaitParity motivation",
    accent: C.red,
    lead: "The mirror response should have a slope of −1. The learned features were far from that rule.",
    bullets: ["Raw pose coordinates contain the constructed signal", "The learned embedding does not decode it", "Sign consistency is only 44.4%"],
    takeaway: "This is a representation failure—not a clinical conclusion.",
    sources: [SRC.late],
    presenter: `We next asked whether the saved representation preserved a simple signed left-right signal. A ridge model was trained in five source-disjoint folds to predict a target constructed directly from pose coordinates. The same target was nearly recoverable from the raw coordinates, so the decoding task itself was feasible.

The learned features did not preserve it: cross-validated R-squared was −0.0678, sign consistency was 0.444, and the response to mirroring had a slope of −0.337 instead of the required −1. This is an informative null result. It does not say that clinically meaningful force asymmetry is impossible; it says that the existing representation should not be presented as if signed anatomy emerged automatically.`,
  }, 10);
  addSimpleSteps(p, {
    title: "A correct mirror changes more than coordinates",
    section: "Parity methodology",
    accent,
    lead: "Reflecting coordinates alone can create an anatomically impossible example.",
    steps: [
      { title: "Flip the axis", body: "Negate the left–right coordinate", accent: C.blue },
      { title: "Swap identities", body: "Exchange every paired left and right landmark", accent: C.red },
      { title: "Swap confidence", body: "Move visibility and confidence with the landmark", accent: C.orange },
    ],
    takeaway: "Applying the mirror twice should recover the original—but real odd/even checks are also required.",
    sources: [SRC.method],
    presenter: `To test parity rigorously, reflection has to be defined as part of the measurement process. We negate the mediolateral coordinate, swap every paired left-right landmark identity, and also swap the corresponding visibility and confidence values. Time, forward motion, and vertical direction remain unchanged.

That last visibility step is easy to miss. If the coordinates are reflected but the missingness pattern stays attached to the old side, the model sees an impossible body and can exploit that shortcut. Applying reflection twice should recover the original input, but that involution test is only a starting check. We also test known odd and even targets on real examples.`,
  }, 11);
  addSimpleRows(p, {
    title: "Three models isolate what internal parity contributes",
    section: "Parity methodology",
    accent,
    lead: "The strongest comparison is not structured versus unstructured—it is structured versus two strong alternatives.",
    rows: [
      { marker: "FIX", markerSize: 20, title: "Output repair", body: "Force the final prediction to flip sign with simple arithmetic.", accent: C.blue },
      { marker: "PAIR", markerSize: 18, title: "Paired, unrestricted", body: "Give the model both views and the same opportunity to interact.", accent: C.orange },
      { marker: "PARITY", markerSize: 18, title: "Equivariant encoder", body: "Require internal even and odd channels to obey the mirror rule.", accent: C.teal },
    ],
    takeaway: "The question is whether organized internal parity beats both alternatives.",
    sources: [SRC.long],
    presenter: `The experiment compares three architectures so that we can identify what, if anything, internal parity contributes. The first is an output-repair baseline: it evaluates the original and mirrored inputs and subtracts the two predictions, which guarantees the final sign flip with a few lines of arithmetic.

The second model processes both views jointly but places no constraint on how their features interact. This controls for the extra view, extra capacity, and cross-view communication. The third model uses the same information but constrains its internal even and odd channels to respond correctly when the branches are exchanged. The co-primary comparisons tell us whether that internal organization adds value beyond either cheap output repair or generic two-view fusion.`,
  }, 12);
  addCollapseGate(p, {
    title: "Perfect symmetry can still hide an empty representation",
    section: "Validation gates",
    accent: C.orange,
    lead: "An odd channel that is always zero satisfies the equation exactly.",
    bullets: ["Does the odd channel vary across people?", "Does it carry more than one trivial direction?", "Do deliberately broken controls fail?"],
    takeaway: "Treat exact parity as a contract test—not proof of useful information.",
    sources: [SRC.long],
    presenter: `An exact transformation rule is not enough, because a useless representation can satisfy it. An odd channel that is always zero passes the equation perfectly: zero is equal to its own negative. We therefore treat exact parity as a contract test, not as evidence of useful information.

Before any downstream interpretation, the audit checks whether the odd channel has measurable energy and variance, whether that variation spans more than one trivial direction, and whether a common offset dominates the signal. We also include positive controls that should fail when the implementation is intentionally broken. These gates prevent an empty or shortcut solution from being mistaken for a scientific result.`,
  }, 13);
  addExperimentOverview(p, {
    title: "The GAVD experiment isolates mirror geometry",
    section: "Controlled experiment",
    accent,
    lead: "Every model sees the same real windows and is compared under two fairness checks.",
    facts: [
      { value: "96", label: "sequences", accent: C.blue },
      { value: "18", label: "source videos", accent: C.teal },
      { value: "394", label: "windows", accent: C.orange },
      { value: "3", label: "paired seeds", accent: "#6D5BD0" },
    ],
    bullets: ["Equal data exposure", "Closely matched compute", "Run-for-run seed pairing", "Parameter counts reported, not hidden"],
    takeaway: "This experiment tests the implementation and audit. It does not test force prediction.",
    sources: [SRC.parity, SRC.long],
    presenter: `The controlled GAVD experiment uses 96 sequences from 18 source videos, producing 394 training windows. Each architecture is trained with seeds 7, 19, and 31. We compare one regime with equal exposure—300 updates and 2,400 example presentations—and another with a closely matched compute proxy of roughly 183 billion operations.

The models are not falsely described as having equal parameter counts: they contain 41,696, 84,256, and 50,272 parameters. Instead, we report those differences and control the comparisons through matched data exposure, closely matched computation, and paired random seeds. This experiment isolates implementation geometry; it does not contain an independent force target.`,
  }, 14);
  addAuditIntuitive(p, {
    title: "The constrained encoder obeys the mirror rule exactly",
    section: "Parity results",
    accent: C.teal,
    lead: "The audit compares the mirror-rule error of the unrestricted control and the equivariant encoder.",
    bullets: ["Equivariant error: exactly 0", "Paired-control error: 5.900", "Health gates pass, but odd-channel energy is small"],
    takeaway: "Exact, non-empty geometry—physical utility still untested.",
    sources: [SRC.parity],
    presenter: `The main result is in panel A. The commutation residual measures the mismatch between two routes: mirror the input and then encode it, or encode it first and then swap the model’s left-right channels. Across both matching regimes, that mismatch is exactly zero for the constrained encoder. The unrestricted two-view control has residuals of 5.900 under equal exposure and 5.286 under matched compute.

The remaining panels test whether the internal channels are alive. The registered gates pass, including positive controls, but the odd-to-even energy ratio is only 0.022 and the effective odd-channel rank is 1.532. So the correct conclusion is exact, nonempty geometry with modest internal diversity—not demonstrated biomechanical utility.`,
  }, 15);
  addTwoColumnBullets(p, {
    title: "Mirror geometry passes; physical utility remains open",
    section: "Discussion",
    accent: C.teal,
    lead: "A correct transformation rule is useful only if it improves a meaningful physical measurement.",
    left: { kicker: "SUPPORTED", title: "Exact, inspectable parity", bullets: ["Worst mirror-rule error is zero", "Strong unrestricted control fails", "All registered health gates pass"], accent: C.teal },
    right: { kicker: "NOT YET SUPPORTED", title: "Meaningful biomechanics", bullets: ["No force target", "No held-out participants", "No demonstrated robustness advantage"], accent: C.red },
    takeaway: "The next experiment must test force on people the model has never fitted.",
    sources: [SRC.parity, SRC.current],
    presenter: `At this point, the supported claim is precise: we have an inspectable model and audit that enforce the intended anatomical reflection rule, survive saving and reloading, handle masks consistently, and distinguish the constrained model from a strong unrestricted control.

Several stronger claims remain open. There is no force target in this experiment, no held-out participant cohort, no demonstrated robustness or label-efficiency advantage, and no GPU replication. The low odd-channel energy and modest rank make the next experiment especially important. We now need to test whether this organization improves a physical measurement, rather than assuming that correct geometry automatically creates useful biomechanics.`,
  }, 16);
  await addPhotoBullets(p, {
    title: "The force study makes parity compete on a physical target",
    section: "Planned study",
    accent: C.orange,
    lead: "Force plates provide the independent target that video alone cannot reveal.",
    image: "polyu_gait_force_lab.jpg",
    imageSide: "right",
    alt: "Participant walking through a university biomechanics laboratory with gait data shown on monitors",
    kicker: "Four-stage study",
    bodyTitle: "Each dataset answers one question",
    bullets: ["AMASS: broad motion pretraining", "BABEL: walking-focused continuation", "Force cohort: held-out participant prediction", "MoVi + sealed cohort: view test and replication"],
    takeaway: "People—not individual gait cycles—are the independent test units.",
    sources: [SRC.planned, SRC.long, "https://www.polyu.edu.hk/bme/research/research-laboratories-and-facilities/", "Hong Kong Polytechnic University Department of Biomedical Engineering photograph"],
    presenter: `The planned study separates the research stages so that one dataset is not asked to support every claim. AMASS provides broad human-motion pretraining. BABEL walking examples narrow the motion prior toward gait without introducing the final clinical outcome. A force-plate cohort then supplies the signed right-minus-left propulsion target, with participants split before any task-specific fitting.

MoVi provides a separate test of real viewpoint stability, and a sealed cohort is reserved for replication after the analysis is frozen. GAVD remains an implementation audit only. For the force claim, the independent observations are people—not individual gait cycles extracted from the same person.`,
  }, 17);
  addSimpleRows(p, {
    title: "The decision rule is fixed before the force results arrive",
    section: "Decision rule",
    accent: C.orange,
    lead: "Primary unit: participant. Primary metric: force error averaged over original and mirrored examples.",
    rows: [
      { marker: "WIN", markerSize: 20, title: "Beats both controls", body: "Internal parity adds measurable value for force prediction.", accent: C.teal },
      { marker: "TIE", markerSize: 20, title: "Ties output repair", body: "Prefer the simpler three-line sign correction for this task.", accent: C.orange },
      { marker: "WAIT", markerSize: 18, title: "Wide or unhealthy result", body: "Call the study inconclusive and collect stronger evidence.", accent: C.red },
    ],
    takeaway: "Secondary measures explain the result; they do not replace the primary force test.",
    sources: [SRC.long, SRC.method],
    presenter: `The outcome table is defined in advance to keep the interpretation honest. The primary unit is the participant, and the primary metric is mean absolute force error averaged over each participant’s original and mirrored examples. The equivariant model must show a meaningful advantage over both the output-repair baseline and the unrestricted two-view model.

If it ties output repair, then the simple arithmetic solution is preferable for this task. If the uncertainty intervals are wide, the result is inconclusive rather than evidence of equivalence. Secondary measures—sign accuracy, calibration, label efficiency, corruption robustness, and view stability—help explain the primary result but do not replace it.`,
  }, 18);
  addFutureIntuitive(p, {
    title: "Biomech-JEPA turns video into a testable physical state",
    section: "Future direction",
    accent,
    lead: "The model should represent body, objects, contact, dynamics, and uncertainty—not one opaque embedding.",
    takeaway: "Qwen chooses which tool to call. Physics tools judge plausibility; they remain the source of physical supervision.",
    sources: [SRC.mentor, SRC.long, SRC.planned],
    presenter: `If GaitParity clears the force-prediction gate, the larger direction is Biomech-JEPA: a world model that represents the body, scene, objects, contacts, dynamics, and uncertainty as related parts of one physical state. That shared state can support both discriminative questions—such as whether a movement is physically grounded—and generative predictions of future body and object keypoints.

Depth estimation, segmentation, and monocular body reconstruction provide additional observations. OpenSim or robot simulation can provide a correction signal during training and flag predictions that violate physical constraints. A Qwen-based policy can learn when an expensive tool call is worth making, but it should route evidence rather than invent biomechanical truth.`,
  }, 19);
  addSimpleRows(p, {
    title: "The collaboration connects physical truth to real-world sensing",
    section: "Research program",
    accent,
    lead: "Each group owns a different part of the measurement problem.",
    rows: [
      { marker: "DELP", markerSize: 19, title: "Biomechanics", body: "Define force targets, OpenSim checks, participant protocols, and meaningful error.", accent: C.teal },
      { marker: "HAI", markerSize: 19, title: "Ambient intelligence", body: "Define viewpoint, depth, occlusion, scene context, and sensing uncertainty.", accent: C.blue },
      { marker: "SHARED", markerSize: 16, title: "World-model core", body: "Build the typed state, JEPA objectives, benchmark, and active tool policy.", accent: "#6D5BD0" },
    ],
    takeaway: "First shared deliverable: PhysMoveBench with a frozen physical and sensing protocol.",
    sources: [SRC.mentor, SRC.planned, SRC.long],
    presenter: `This research program naturally combines three kinds of expertise. The Delp biomechanics lab can define the physical targets, force-plate protocol, OpenSim comparisons, and clinically meaningful error measures. The HAI ambient intelligence lab can define realistic variation in viewpoint, depth, scene context, occlusion, and uncertainty. The world-model effort connects those measurements through an auditable representation and an active tool-use policy.

I see this as a proposed structure rather than a fixed division of labor. A strong first shared deliverable would be PhysMoveBench: one benchmark where force, contact, viewpoint, occlusion, and uncertainty are evaluated under a frozen protocol. That gives the collaboration a common measurement language before scaling the model.`,
  }, 20);
  addConclusionIntuitive(p, {
    title: "GaitParity is the first gate—not the final system",
    section: "Conclusion",
    accent,
    statement: "The project now has a precise mirror rule and a controlled way to test whether that structure becomes useful.",
    bullets: ["Video alone does not reveal physical causes", "The equivariant encoder passes the mirror audit exactly", "Held-out force prediction is the next decisive test"],
    bottom: "If the force study succeeds, Biomech-JEPA becomes a credible path toward ambient physical movement understanding.",
    sources: [SRC.current, SRC.planned, SRC.mentor],
    presenter: `The project began with a simple gap: observing a skeleton is not the same as understanding the physical process that produced it. GaitParity now gives us a precise anatomical rule, an implementation that satisfies that rule exactly, a control that fails for the expected reason, and health checks that rule out an empty odd channel.

The next step is to earn the stronger claim with held-out people, force plates, real viewpoint changes, and independent replication. The long-term goal is not to replace physics with a language model. It is to learn an efficient physical state estimator whose symmetry, uncertainty, and tool calls remain measurable and testable.`,
  }, 21);
  return p;
}

async function saveDeck(presentation, filename) {
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(path.join(OUT, filename));
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  if (process.argv[2] === "technical-only") {
    const technical = await buildTechnical();
    await saveDeck(technical, "GaitParity_Technical_Audience.pptx");
    console.log("Updated the technical GaitParity deck.");
    return;
  }
  if (process.argv[2] === "technical-intuitive") {
    const technical = await buildTechnical();
    await saveDeck(technical, "GaitParity_Technical_Audience_Intuitive.pptx");
    console.log("Created the intuitive technical GaitParity deck.");
    return;
  }
  const [hs, general, technical] = await Promise.all([buildHighSchool(), buildGeneral(), buildTechnical()]);
  await saveDeck(hs, "GaitParity_Advanced_High_School.pptx");
  await saveDeck(general, "GaitParity_General_Audience.pptx");
  await saveDeck(technical, "GaitParity_Technical_Audience.pptx");
  console.log("Created three GaitParity decks.");
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
