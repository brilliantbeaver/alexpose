#!/usr/bin/env python3
"""Draw the editable vector illustration used by the GAVD media reports."""
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT = Path(__file__).resolve().parent / "assets/gavd-sequence-extraction.svg"


def build_svg():
    cards = [
        (40, "#2563eb", "#eff6ff", "1 · Read the CSV", ["Read the condition, YouTube ID,", "and first and last frame numbers."], "file"),
        (324, "#7c3aed", "#f5f3ff", "2 · Save the source", ["Download the full video into", "the matching condition folder."], "download"),
        (608, "#0d9488", "#f0fdfa", "3 · Keep the range", ["Keep every listed frame, including", "the first and the last frame."], "film"),
        (892, "#ea580c", "#fff7ed", "4 · Check every frame", ["Compare each saved frame with", "its place in the downloaded video."], "check"),
    ]
    icons = {
        "file": '<path d="M15 9h21l10 10v34H15z"/><path d="M36 9v12h10M23 31h16M23 39h16M23 47h10"/>',
        "download": '<path d="M30 10v28m-10-10 10 10 10-10M13 42v10h34V42"/>',
        "film": '<rect x="10" y="14" width="40" height="34" rx="3"/><path d="M18 14v34M42 14v34M10 25h8M10 37h8M42 25h8M42 37h8"/><path d="m27 24 10 7-10 7z"/>',
        "check": '<path d="M30 9 49 17v15c0 12-19 23-19 23S11 44 11 32V17z"/><path d="m21 31 7 7 13-15"/>',
    }
    fragments = []
    for x, color, tint, heading, lines, icon in cards:
        fragments.append(f'''<g transform="translate({x} 146)">
          <rect width="260" height="218" rx="23" fill="white" filter="url(#shadow)"/>
          <rect x="19" y="20" width="61" height="61" rx="18" fill="{tint}"/>
          <g transform="translate(20 20)" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">{icons[icon]}</g>
          <text x="22" y="117" fill="#14213d" font-size="19.5" font-weight="700">{heading}</text>
          <text x="22" y="151" fill="#475569" font-size="14.2">{lines[0]}</text>
          <text x="22" y="174" fill="#475569" font-size="14.2">{lines[1]}</text>
          <rect x="22" y="197" width="48" height="4" rx="2" fill="{color}"/>
        </g>''')
    for x in (306, 590, 874):
        fragments.append(f'<path d="M{x} 256h10" stroke="#94a3b8" stroke-width="2.8" stroke-linecap="round" marker-end="url(#arrow)"/>')
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title description">
      <title id="title">How a GAVD sequence becomes a video clip</title>
      <desc id="description">Four steps read the CSV, download the source video, keep the inclusive frame range, and compare every saved frame. The example keeps frames sixteen through one hundred eleven, producing ninety-six frames.</desc>
      <defs>
        <linearGradient id="background" x2="1" y2="1"><stop stop-color="#f2f7ff"/><stop offset="1" stop-color="#f3fdf9"/></linearGradient>
        <linearGradient id="selection" x2="1"><stop stop-color="#14b8a6"/><stop offset="1" stop-color="#10b981"/></linearGradient>
        <filter id="shadow" x="-15%" y="-15%" width="140%" height="150%"><feDropShadow dx="0" dy="7" stdDeviation="11" flood-color="#304e80" flood-opacity=".09"/></filter>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="4.5" refY="3" orient="auto"><path d="m1 1 3 2-3 2" fill="none" stroke="#94a3b8" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></marker>
      </defs>
      <rect width="1200" height="760" rx="28" fill="url(#background)"/>
      <circle cx="1152" cy="14" r="143" fill="#ddd6fe" opacity=".35"/>
      <circle cx="39" cy="738" r="137" fill="#99f6e4" opacity=".28"/>
      <g font-family="Inter, Arial, Helvetica, sans-serif">
        <text x="42" y="60" font-size="30" font-weight="750" fill="#14213d">How a GAVD sequence becomes a video clip</text>
        <text x="43" y="95" font-size="17" fill="#52627a">The CSV tells us which video to use and exactly which frames to keep.</text>
    ''' + "".join(fragments) + '''
        <rect x="40" y="395" width="1112" height="282" rx="25" fill="white" filter="url(#shadow)"/>
        <text x="64" y="434" font-size="21" font-weight="700" fill="#14213d">The normal example keeps 96 frames.</text>
        <text x="65" y="462" font-size="15" fill="#52627a">We keep frames 16 through 111, including both ends of the range.</text>
        <rect x="75" y="490" width="1060" height="62" rx="13" fill="#e9eef5"/>
        <rect x="280" y="490" width="650" height="62" fill="url(#selection)"/>
        <path d="M280 481v81M930 481v81" stroke="#0f766e" stroke-width="2.5" stroke-dasharray="4 4"/>
        <text x="177" y="527" text-anchor="middle" font-size="15" fill="#64748b">Earlier frames</text>
        <text x="605" y="527" text-anchor="middle" font-size="23" font-weight="700" fill="white">Keep frames 16 through 111.</text>
        <text x="1033" y="527" text-anchor="middle" font-size="15" fill="#64748b">Later frames</text>
        <text x="280" y="584" text-anchor="middle" font-size="14" font-weight="650" fill="#0f766e">Frame 16 is included.</text>
        <text x="930" y="584" text-anchor="middle" font-size="14" font-weight="650" fill="#0f766e">Frame 111 is included.</text>
        <rect x="75" y="610" width="1060" height="42" rx="11" fill="#eef2ff"/>
        <text x="605" y="637" text-anchor="middle" font-size="16.5" fill="#4338ca">Save the selected frames as normal/3KnFt8bH3tE-16-111.mp4.</text>
        <text x="600" y="718" text-anchor="middle" font-size="15" fill="#52627a">Each clip keeps the full image from the download. Audio and chapter information are left out.</text>
      </g>
    </svg>'''


def main():
    svg = build_svg()
    ET.fromstring(svg)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
