"""Produce the agent-reviewed bibliography audit from stored metadata.

Overrides below are transcribed from primary pages/manuscripts that were opened
and inspected, not fuzzy matches. Candidate records with a poor title match are
never promoted to verified references.
"""
from pathlib import Path
import html, json, re

ROOT = Path(__file__).parent
rows = json.loads((ROOT / 'reference-crossref-candidates.json').read_text())
overrides = {
28: dict(authors='Valentin Bazarevsky; Ivan Grishchenko; Karthik Raveendran; Tyler Zhu; Fan Zhang; Matthias Grundmann', year=2020, venue='arXiv:2006.10204', url='https://arxiv.org/abs/2006.10204'),
31: dict(authors='Yuxin Chen; Ziqi Zhang; Chunfeng Yuan; Bing Li; Ying Deng; Weiming Hu', year=2021, venue='ICCV, 13359–13368', url='https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html'),
33: dict(authors='Wentao Zhu; Xiaoxuan Ma; Zhaoyang Liu; Libin Liu; Wayne Wu; Yizhou Wang', year=2023, venue='ICCV, 15085–15099', url='https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html'),
35: dict(authors='Mathilde Caron; Hugo Touvron; Ishan Misra; Hervé Jégou; Julien Mairal; Piotr Bojanowski; Armand Joulin', year=2021, venue='ICCV, 9650–9660', url='https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html'),
39: dict(authors='Alexei Baevski; Wei-Ning Hsu; Qiantong Xu; Arun Babu; Jiatao Gu; Michael Auli', year=2022, venue='ICML, PMLR 162, 1298–1312', url='https://proceedings.mlr.press/v162/baevski22a.html'),
40: dict(authors='Adrien Bardes; Jean Ponce; Yann LeCun', year=2022, venue='ICLR 2022; arXiv v3 explicitly identifies accepted venue', url='https://arxiv.org/abs/2105.04906'),
42: dict(authors='Adrien Bardes; Quentin Garrido; Jean Ponce; Xinlei Chen; Michael Rabbat; Yann LeCun; Mahmoud Assran; Nicolas Ballas', year=2024, venue='Transactions on Machine Learning Research', url='https://arxiv.org/abs/2404.08471'),
55: dict(authors='Niki Martinel; Mariano Serrao; Christian Micheloni', year=2024, venue='arXiv:2411.19544', url='https://arxiv.org/abs/2411.19544'),
}
notes = {
1:'Clinical MS gait background. It does not validate the particular 12-landmark sampling prior.',
2:'Bibliographic match. Removed from streamlined revision; no additional specific neuroanatomical inference adopted.',
3:'Bibliographic match. Removed; no somatosensory mechanism inferred from this video classifier.',
4:'2020 is the volume/issue year; online publication was 2019. Original §6.1 misuses this spasticity paper as the effective-rank citation.',
5:'Supports MS gait asymmetry/coordination background; current classifier does not directly estimate those clinical measures.',
6:'Bibliographic match; instrumented-walkway study, not validation of web-video disease labels.',
7:'Bibliographic match; no further specific clinical claim retained.',
8:'Bibliographic match; clinical GAITRite work cannot establish physical validity of normalized image coordinates.',
9:'2016 online and issue year agree; no further specific clinical claim retained.',
10:'Bibliographic match; a systematic review, not a primary experiment for direct numerical comparison.',
11:'Bibliographic match; gait-dynamics background does not explain this model’s confusion errors.',
12:'Bibliographic match; no freezing-of-gait measurement was made in the current experiment.',
13:'Bibliographic match; article number 459321 (Crossref also records pages 1–8).',
14:'Bibliographic match; 2024 publication is real, not a future or fabricated citation.',
15:'Modality mismatch in original related work: smartphone inertial/accelerometer monitoring, not camera pose estimation. Primary author manuscript confirms this.',
16:'Video gait study with prospective validation and specific camera calibration; does not establish equivalent validity for this collection.',
17:'Bibliographic match; relevant PD video-assessment context, but not a matched classification baseline.',
18:'Bibliographic match including Katsuki Eguchi and 2024 journal article. PD versus SCD feasibility differs from unverified Normal/MS/PD web labels.',
19:'Online 2025-12-13; assigned volume 9, article 98 (2026). Original 2026 citation is legitimate. Neither date should be silently called fabricated.',
20:'Bibliographic match; broader clinical movement measurement differs from the present collection-label prediction.',
21:'Bibliographic match; multi-camera smartphone workflow is not a validation of this single-camera normalization.',
22:'Directly supports validated video gait estimation under the paper’s acquisition conditions; not automatic portability to web videos.',
23:'Directly supports clinical/perspective validation context; avoid claiming that it clinically validates this classifier.',
24:'Bibliographic match; treadmill validity is not general outdoor-web-video validity.',
25:'Bibliographic match; review/background source, not a direct comparison experiment.',
26:'2021 is the TPAMI issue year; DOI/online year 2019 is not an error. Original title and author initials agree.',
27:'Original CVF pages 5693–5703 verified; Crossref/IEEE pages 5686–5696 differ. Use the cited publisher edition consistently.',
28:'Preprint exists with all six stated authors and 2020 date. Bad Crossref search match was discarded.',
29:'Bibliographic match; skeleton action recognition, not disease classification or guaranteed privacy.',
30:'Original CVF pages 12026–12035 agree with primary PDF; Crossref/IEEE has 12018–12027.',
31:'Full author names verified at CVF. Generic action recognition, not clinical validation.',
32:'Original CVF pages 2969–2978 verified; Crossref/IEEE has 2959–2968.',
33:'Full author names/title/year/pages verified at CVF; 3D lifting pretraining differs from the current 2D+visibility objective.',
34:'Original calls this a transformer-based SkeletonMAE, but this cited paper uses a graph-based encoder-decoder. CVF pages 5606–5618 differ from IEEE 5583–5595.',
35:'Original §7.2 wrongly labels DINO as a JEPA study. Bibliographic entry itself is real and correct.',
36:'Original §7.2 wrongly labels MAE as a JEPA study. CVF pages 16000–16009 differ from IEEE 15979–15988.',
37:'Bibliographic match; VideoMAE is masked reconstruction, not a source of clinical gait evidence.',
38:'MaskFeat studies several prediction targets, particularly HOG; avoid asserting learned semantic latent targets as its sole mechanism. CVF pages 14668–14678 differ from IEEE 14648–14658.',
39:'Verified at PMLR. The poor Crossref match to data2vec-SG was rejected; it is a different paper.',
40:'Author list/title and ICLR 2022 venue verified in primary PDF; arXiv first submission 2021 is not the conference year. Poor Crossref match rejected.',
41:'Original §2.2 incorrectly cites this image JEPA for S-JEPA masking. Bibliographic entry itself is correct.',
42:'Original §2.2 incorrectly cites this video JEPA for S-JEPA masking. TMLR acceptance is confirmed by the journal’s dated announcement; OpenReview HTML access was challenged. Primary manuscript uses Mahmoud; journal announcement uses Mido for Assran.',
43:'2018 is the NeuroImage volume year; DOI/online publication 2017 is not an error. Supports dependent-fold/small-sample caution.',
44:'Bibliographic match; motivates selection/test separation, not a guarantee that this pipeline is leakage-free.',
45:'Bibliographic match; gait identity recognition rather than clinical diagnosis.',
46:'Original CVF pages 1569–1577 verified; Crossref/IEEE has 1568–1576. Gait identity task.',
47:'Original CVF pages 19595–19604 agree with primary PDF; Crossref/IEEE has 19538–19547. Cross-domain identity recognition does not establish clinical transfer.',
48:'Bibliographic match; identity recognition. Omitted from streamlined revision.',
49:'Bibliographic match; original “closest related work” ranking was unsupported as an exhaustive literature judgment.',
50:'Bibliographic match; original author initials agree with full names in deposited metadata.',
51:'Bibliographic match; original author initials agree with full names in deposited metadata.',
52:'Bibliographic match; 2017 issue year, DOI contains 2016 online year.',
53:'Bibliographic match; clinical arm-swing review does not validate the particular masking prior.',
54:'Title/authors/ACM 2023 pages verified. §3.2 of primary paper uses body-part topology masking. Original “more recent” framing after S-JEPA 2024 is chronologically wrong.',
55:'Preprint exists. First author is Niki Martinel (original N. is valid), not Nicola. No peer-reviewed venue should be invented.',
56:'Bibliographic match; 2021 paper has a DOI containing 2020. Review evidence, not primary experiment for matched comparison.',
57:'Correct authors are Mohamed Abdelfattah and Alexandre Alahi. ECCV 2024 manuscript/online 2024; Springer print metadata can show 2025. Masking preferentially samples motion, not deterministic top-motion selection.',
}
cvf = {
27:'https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html',
30:'https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf',
32:'https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html',
34:'https://openaccess.thecvf.com/content/ICCV2023/html/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.html',
36:'https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html',
38:'https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf',
46:'https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/html/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper.html',
47:'https://openaccess.thecvf.com/content/ICCV2023/papers/Fu_GPGait_Generalized_Pose-based_Gait_Recognition_ICCV_2023_paper.pdf',
}

intro = '''# Audit of all 57 original references

Verified 2026-10-08. Scope: every numbered reference in the supplied `wacv2007-v1.pdf`, including references omitted from the revised paper. No entry was accepted from a fuzzy search match alone. All 57 cited works could be located; **no fabricated work or incompatible original author list was identified**. Important defects instead concern citation-to-claim mismatches and inconsistent interpretation of publication dates. This is not a certification of every scientific conclusion in each cited paper.

The table gives the canonical title, full author names and bibliographic year from primary manuscripts, publisher/conference pages, or publisher-deposited Crossref metadata. Original author initials and listed authors were compared with these records. Where the original uses “et al.”, omitted coauthors are expanded here without treating abbreviation as an error. Capitalization, curly apostrophes and hyphen variants are normalized for matching. Publisher metadata and explicit primary-source overrides are preserved in `reference-crossref-candidates.json` and the script `build_reference_audit.py`; original strings are in `reference-originals.json`. Search candidates with HTTP failures or poor matches remain in the evidence JSON as failed searches, never as verification.

**Status** is “Verified” for bibliographic identity and “Verified; claim correction” where a known original assertion needs correction. “Verified” does not mean a paper clinically validates the present experiment. Specific claim checks and remaining scope limits are recorded in the last column. For omitted background sources, bibliographic checking was completed; an exhaustive full-text replication of their findings was not performed. All retained scientific claims must use the narrower support map in `venue-and-literature.md`.

Several CVF and IEEE/Crossref page ranges differ. The original generally used legitimate CVF pagination; these are edition differences, not hallucinations. Dates likewise distinguish preprint, online and issue/proceedings assignment. In particular, [19] genuinely has a 2025 online date and 2026 issue assignment, and [57] is ECCV 2024 despite Springer print metadata listing 2025.

| Original no. | Status | Verified title and full authors | Year / venue record | Primary record and audit finding |
|---|---|---|---|---|
'''
lines = [intro]
structured = []
claim_corrections = {4,15,34,35,36,38,41,42,49,54,57}
for r in rows:
    n=r['number']
    c=(r.get('candidates') or [{}])[0]
    m=r.get('manual_doi_metadata') or c.get('metadata',{})
    if n in overrides:
        o=overrides[n]
        authors=o['authors']; year=o['year']; venue=o['venue']; url=o['url'];title=r['title']
    else:
        assert m and (r.get('manual_doi_metadata') or c['similarity']>.98), (n,c.get('similarity'))
        title=html.unescape(m['title'][0]).replace('\xa0',' ')
        authors='; '.join((a.get('given','')+' '+a.get('family','')).strip() for a in m.get('author',[]))
        year=(m.get('published-print') or m.get('published') or m.get('issued'))['date-parts'][0][0]
        if n==19:year=2026
        if n==57:year=2024
        venue=html.unescape('; '.join(m.get('container-title',[])))
        if m.get('volume'):venue+=' '+str(m['volume'])
        if m.get('issue'):venue+='('+str(m['issue'])+')'
        p=m.get('article-number') or m.get('page')
        if p:venue+=', '+str(p)
        url='https://doi.org/'+m['DOI']
    status='Verified; claim correction' if n in claim_corrections else 'Verified'
    support='[Primary record]('+url+')'
    if n in cvf:support+='; [CVF edition]('+cvf[n]+')'
    if n==42:support+='; [TMLR announcement](https://groups.google.com/g/tmlr-announce-weekly/c/Bs02CFGTU6g)'
    if n==15:support+='; [author manuscript](https://www.andrewcreagh.com/assets/pdf/Creagh_2022_longitudinal-trend-monitoring.pdf)'
    lines.append(f'| {n} | {status} | **{title}**<br>{authors} | {year}; {venue} | {support}. {notes[n]} |\n')
    structured.append(dict(number=n,status=status,title=title,authors=authors,year=year,venue=venue,url=url,note=notes[n]))
lines.append('''
## Corrections carried into revised writing

The revised bibliography uses verified primary sources with explicit BibTeX keys. The supplied S-JEPA [57] author names are correct; replacing them would have created an error. The effective-rank reference [4] is replaced by Roy and Vetterli (2007). S-JEPA masking is attributed to [57], rather than I-JEPA/V-JEPA [41,42]. DINO and MAE [35,36] are no longer called JEPA studies. Yan et al.’s SkeletonMAE [34] is described as graph-based. Creagh et al. [15] is not used as camera-validation evidence. The untested clinical landmark bias remains a heuristic rather than a literature-proven diagnostic mask.

No source was silently removed to conceal a failed existence check. Removed sources remain in this audit. No direct quotations from their abstracts are incorporated into the revised manuscript; technical claims are narrow paraphrases tied to the relevant primary studies.
''')
(ROOT/'reference-audit.md').write_text(''.join(lines))
(ROOT/'reference-audit.json').write_text(json.dumps(structured,indent=2,ensure_ascii=False))
print('Wrote audit for',len(structured),'references; statuses:',{x['status'] for x in structured})
