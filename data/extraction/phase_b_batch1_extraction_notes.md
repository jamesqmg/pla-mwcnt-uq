# Phase B Batch 1 — Extraction Notes

**Date:** 2026-04-27
**Extractor:** automated PDF → pymupdf4llm → schema extraction
**Batch scope:** 5 FDM/TSE-prioritized open-access papers from Phase A candidate list
**Papers successfully extracted:** 4/5
**Papers requiring manual PDF download:** 0/5
**Validation pass completed:** 2026-04-27 (3 new papers extracted)

---

## Status summary

| Paper | DOI | PDF retrieved | Extracted | Blocker |
|---|---|---|---|---|
| Mohammed et al. (2025) — IJE | 10.5829/ije.2025.38.08b.05 | ✓ | ✓ | — |
| [2025] PLA-MWCNT filaments (Manufacturing Letters) | 10.1016/j.mfglet.2025.07.004 | ✗ WRONG PDF | ✗ | PDF retrieved (1-s2.0-S2213846325002123-main.pdf) is a DIFFERENT Manufacturing Letters article about Ti6Al4V blisk machining (Seimann et al.). Same journal issue, wrong article. Needs correct PDF. |
| [2018] MWCNTs/PLA 3D — Micromachines | 10.3390/mi9120635 | ✓ | ✓ | Extracted — see Paper 2 below |
| [2024] FFF process optimisation — Polymer Eng & Sci | 10.1002/pen.26900 | ✓ | ✓ | Extracted — see Paper 3 below |
| [2025] Annealing PLA/CNT — Polymers | 10.3390/polym17020246 | ✓ | ✓ PARTIAL | Extracted — see Paper 4 below |

**Manual download URLs** (open these in your browser while logged in via Mapúa institutional access or directly from the publisher):
- `https://www.mdpi.com/2072-666X/9/12/635/pdf` (Micromachines 2018, MDPI OA) — ✓ DOWNLOADED
- `https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/pen.26900` (Polymer Eng & Sci 2024, Wiley) — ✓ DOWNLOADED
- `https://www.mdpi.com/2073-4360/17/2/246/pdf` (Polymers 2025, MDPI OA) — ✓ DOWNLOADED
- `https://www.sciencedirect.com/science/article/pii/S2213846325002123` (Manufacturing Letters 2025, Elsevier) — ✗ WRONG PAPER RETRIEVED (see status table)

**ACTION NEEDED:** The correct Manufacturing Letters 2025 paper (DOI: 10.1016/j.mfglet.2025.07.004) is about PLA/MWCNT FFF filaments. The downloaded PDF is a blisk machining paper by Seimann et al. that happened to appear in the same journal issue. Download the correct article directly via DOI or Elsevier institutional access.

---

## Paper 1: Mohammed et al. (2025) — EXTRACTED ✓

**Citation:** Mohammed OK, Hassani A, Zolfaghari A, Diwan AA. Mechanical Performance of 3D Printed Polylactic Acid/MWCNT Brackets: Effects of CNT Concentration and Printing Orientation on Stress-Strain Behavior and Deflection Characteristics. *International Journal of Engineering, Transactions B: Applications.* 2025;38(08):1782–1792.

**DOI:** 10.5829/ije.2025.38.08b.05
**Output file:** `per_paper/mohammed_2025_ije.csv`
**Rows extracted:** 8 (4 wt% levels × 2 print orientations)

### Extraction decisions

**Tensile data source:** Table 3 (numerical). Data is NOT graphical-only — explicit MPa values in a formatted table. No GRAPHICAL ONLY flag needed.

**Matrix control values:** Table 3 includes 0 wt% MWCNT rows for both 0° and 90° orientations. Used as the matrix tensile strength baseline per paper. Note: Table 1 lists "Tensile strength (MPa): 60" for PLA from reference [29] — this is a literature value, NOT the paper's own measurement. The actual control specimens measured at 43.786 MPa (0°) and 14.615 MPa (90°). Used measured values, not literature values.

**Print orientation FLAG:** This paper tests two print orientations (0° and 90°) at each wt% level. The Hiremath schema has no "print orientation" column. Both orientations are extracted as separate rows because:
1. They produce substantially different tensile values (e.g., 57.9 MPa vs 22.4 MPa at 3 wt%, 0° vs 90°)
2. Dropping one orientation would discard valid data points
3. Including both inflates within-wt% variance in a way that may not be representative of single-orientation FDM composites

**Decision deferred to student:** Whether to include 0° only, 90° only, both, or average. Recommend: use 0° only (loading-direction aligned = standard tensile specimen convention) unless adviser instructs otherwise. The 0° rows are closer to what ASTM D638 FDM specimens typically report in the literature.

**Inferred values (flagged in CSV):**
- Matrix density: 1.23 g/cm³ — midpoint of Table 1 range (1.21–1.25). Range is from reference [29], not measured by authors. Use with caution; actual printed density may differ.
- CNT diameter: 35 nm — midpoint of Table 2 range (20–50 nm). Supplier spec.
- CNT length: 22.5 μm — midpoint of Table 2 range (10–35 μm). Supplier spec.
- CNT Young's modulus: 1.0 TPa — standard MWCNT default; not reported in paper.

**Testing standard:** ASTM D638 Type I explicitly stated in methods.

**Processing route:** FDM/3D Printing confirmed. Composite filament fabricated via melt blending → single-screw extrusion (1.75 mm diameter), then FDM printed. The Hiremath vocabulary "3D Printing" maps correctly.

**CNT modification:** Pristine assumed (no functionalization mentioned; text describes CNTs as received from Sky Spring Nanomaterials Inc. without surface treatment).

**Quality assessment:** HIGH. Table 3 is clean, complete, and unambiguous. All four columns (wt%, tensile MPa, control MPa, Young's modulus) are numeric. The only deferred decision is print orientation handling.

---

## Papers 2–5: Pending manual download

See status table above. Once PDFs are downloaded and placed in `data/pdfs/`, re-run:

```bash
cd "/Users/jimbo/Downloads/MAPUA/Thesis 1"
python3 - << 'EOF'
import pymupdf4llm
from pathlib import Path
for pdf in Path("data/pdfs").glob("*.pdf"):
    md = pymupdf4llm.to_markdown(str(pdf))
    md_path = Path("data/extraction") / (pdf.stem + ".md")
    md_path.write_text(md, encoding="utf-8")
    print(f"Converted: {md_path.name}  ({len(md)//1000}K chars)")
EOF
```

Then run the schema extraction on each converted markdown.

---

---

## Paper 2: Luo et al. (2018) — Micromachines — EXTRACTED ✓

**Citation:** Luo J, Wang H, Zuo D, Ji A, Liu Y. Research on the Application of MWCNTs/PLA Composite Material in the Manufacturing of Conductive Composite Products in 3D Printing. *Micromachines.* 2018;9(12):635.

**DOI:** 10.3390/mi9120635
**Output file:** `per_paper/luo_2018_micromachines.csv`
**Rows extracted:** 5 (0, 1, 3, 5, 7 wt% MWCNT)

### Extraction decisions

**Tensile data source:** Table 1 (numeric). Values given as mean ± SD. Used mean values. n=6 per group.

**Young's Modulus units in table:** Paper reports Young's Modulus in MPa (range 78.4–134.4 MPa). This is unusually low for a modulus — likely reflects a low-compliance test configuration (5 N load cell at 2 mm/min on FDM specimens) rather than bulk material modulus. Converted to GPa in CSV (divide by 1000). Flag: adviser should verify whether this is a true modulus or a chord stiffness value.

**Testing standard:** NOT specified. Paper names the machine (Instron 5567) and test speed (2 mm/min, 5 N load cell) but cites no ASTM or ISO standard. Listed as "unspecified."

**CNT specs:** Diameter 7–15 nm (midpoint 11 nm), length 5–10 mm (midpoint 7.5 mm = 7500 µm). The length value is unusual — most MWCNT suppliers report length in µm, not mm. If paper's "5–10 mm" is correct, these are exceptionally long CNTs. If it's a typo for 5–10 µm, that is more typical. Flag this discrepancy; use paper's stated value (7500 µm midpoint) with note.

**Processing route:** Solution mixing → twin-screw extrusion → FDM printing. Confirms FDM as the final fabrication method.

**CNT modification:** Pristine — no surface treatment mentioned.

**9 wt% condition:** Not extractable — paper reports 9% was not printable (wire breakage during filament fabrication). Not included.

**Quality gate: PARTIAL** — tensile data directly from Table 1 (numeric), wt% explicit, n=6 stated. Fails PASS threshold because testing standard is unspecified. PLA-filtered relevance: HIGH (PLA + MWCNT + FDM).

---

## Paper 3: Hardani et al. (2024) — Polymer Engineering & Science — EXTRACTED ✓

**Citation:** Hardani H, Afshari M, Allahyari F, Samadi MR, Afshari H, Medina EMM. Optimization of FFF process parameters to improve the tensile strength and impact energy of polylactic acid/carbon nanotube composite. *Polymer Engineering & Science.* 2024;64(10):5047–5060.

**DOI:** 10.1002/pen.26900
**Output file:** `per_paper/hardani_2024_pen.csv`
**Rows extracted:** 15 (full RSM Box–Behnken design — all 15 experiments)

### Extraction decisions

**Tensile data source:** Table 3 (design matrix — numeric). All 15 experiments have explicit CNT content (wt%), print speed, nozzle temperature, tensile strength (MPa), and impact energy (J). Clean numeric table — no inference needed.

**CNT type:** Paper calls them "CNTs" from INP Corporation (9.5 nm diameter, 1.5 µm length, >90% purity, specific surface area 250–300 m²/g). These specs are consistent with MWCNT geometry (small diameter, high surface area) but the paper never uses the term "MWCNT." Listed as "unspecified" in CNT_type with note in Extraction Notes column.

**Process variables:** This paper introduces two process parameters (print speed, nozzle temperature) that are not in the Hiremath schema. These are captured in the Extraction Notes column. For Hiremath-schema comparability, the relevant variable is CNT wt% — three levels tested (0, 2, 4 wt%) across the 15 RSM runs.

**Matrix tensile strength:** Supplier spec from materials section = 62 MPa (ASTM D638). Used as matrix reference; actual measured control values range 39.36–46.29 MPa depending on print conditions.

**Testing standard:** ASTM D638 explicitly stated. Specimens printed via FFF.

**Optimal condition row:** Not extracted as a separate row — the optimization result (2.8 wt% CNT, 20 mm/s, 209°C → 62.5 MPa predicted / 63.9 MPa verified) is a modeled prediction, not a direct measurement from Table 3. Could be added as an inferred row if advisers want optimum-condition data point.

**Quality gate: PASS** — tensile from clean numeric Table 3, wt% explicit, ASTM D638 stated. Only caveat is CNT type ambiguity. PLA-filtered relevance: HIGH (PLA + FFF + ASTM D638, explicit MWCNT-like CNT).

---

## Paper 4: da Silva et al. (2025) — Polymers — EXTRACTED ✓ PARTIAL

**Citation:** da Silva FU, Luna CBB, da Silva FS, Barreto JVM, Schmitz DP, Soares BG, Wellen RMR, Araújo EM. Exploring the Effect of Annealing on PLA/Carbon Nanotube Nanocomposites: In Search of Efficient PLA/MWCNT Nanocomposites for Electromagnetic Shielding. *Polymers.* 2025;17(2):246.

**DOI:** 10.3390/polym17020246
**Output file:** `per_paper/dasilva_2025_polymers.csv`
**Rows extracted:** 5 (PLA + 1/2/3/4 phr MWCNT, non-annealed only)

### Extraction decisions

**Tensile data source:** MIXED — Figure 12b is graphical (bar chart, values not tabulated). However, the text in Section 3.6 and the Abstract provide specific numeric values: pure PLA = 63.2 MPa, 1 phr = ~63.2 MPa (text: "did not affect tensile strength"), 2 phr = 65.1 MPa (abstract explicit), 3–4 phr = "64.8–65.2 MPa range" (text). Used these text/abstract values; flag as PARTIAL because most values come from prose not a table.

**Annealing rows:** EXCLUDED from CSV. Annealing is a secondary post-processing treatment not represented in the Hiremath schema. Annealed tensile values (56.5–65.3 MPa) are available if needed but kept separate. A separate annealing-effects CSV could be generated if advisers want to explore this.

**phr to wt% conversion:** phr (parts per hundred resin) converted to wt%: wt% = phr / (100 + phr) × 100. At these low concentrations the difference is small (1 phr = 0.99 wt%, 4 phr = 3.85 wt%).

**Processing route:** Twin-screw extrusion (ZSK, D=18mm, L/D=40) + injection molding. NOT FDM. This reduces relevance to the thesis filtered dataset.

**Testing standard:** ASTM D638 explicitly stated. n=10 per group.

**MWCNT specs:** Outer diameter 8–15 nm (midpoint 11.5 nm), length 10–50 µm (midpoint 30 µm), purity >99%, CVD method, Yuechuang Technology.

**Quality gate: PARTIAL** — tensile strength for 2 phr explicitly stated in abstract (PASS-level), but 1/3/4 phr values only from prose ranges (not tabulated). Testing standard confirmed. Processing is injection molding, not FDM. PLA-filtered relevance: MEDIUM (PLA + MWCNT, non-FDM processing).

---

## Paper 5: Seimann et al. (2025) — Manufacturing Letters — FAILED ✗

**DOI retrieved:** 10.1016/j.mfglet.2025.07.004 (intended — PLA/MWCNT filaments)
**Actual PDF content:** "Digital process chain: Scientific based manufacturing of an intake fan blisk prototype" by Seimann M et al., about Ti6Al4V 5-axis milling, Fraunhofer IPT. Zero relevance to CNT composites.
**Blocker:** Wrong article retrieved — same journal issue, different paper. The Elsevier URL or download may have served an adjacent article from the conference proceedings volume.
**Action required:** Retrieve correct article via DOI 10.1016/j.mfglet.2025.07.004 directly.

---

---

## Paper 6: Vidakis et al. (2021) — Nanocomposites — EXTRACTED ✓

**Citation:** Vidakis N, Petousis M, Kourinou M, Velidakis E, Mountakis N, Fischer-Griffiths PE, Grammatikos S, Tzounis L. Additive manufacturing of multifunctional polylactic acid (PLA)—multiwalled carbon nanotubes (MWCNTs) nanocomposites. *Nanocomposites.* 2021;7:1, 184–199.

**DOI:** 10.1080/20550324.2021.2000231
**Output file:** `per_paper/vidakis_2021_taylorandfrancis.csv`
**Rows extracted:** 6 (0, 0.1, 0.5, 1.0, 2.5, 5.0 wt% MWCNT)

### Extraction decisions

**Tensile data source:** Figure 3(b) bar chart — numeric labels visible directly on bars (exact values: 45.7, 42.6, 46.9, 61.3, 59.4, 58.6 MPa) with error bars labeled. Young's modulus from Figure 3(c) (exact labels: 284.3, 252.3, 298.4, 380.1, 360.3, 358.7 MPa → converted to GPa).

**Processing route:** FFF (FDM) confirmed — Intamsys Funmat HT printer, 0.4 mm nozzle. Filament produced by melt extrusion (3D Evo single screw extruder) then FFF printed. Specimens tested are 3D-printed parts, NOT filaments.

**Testing standard:** ASTM D638-02a explicitly stated. Type V specimen (3.2 mm thickness). 5 specimens per group. 10 mm/min crosshead speed. Printing orientation 45° off-axis per methods.

**MWCNT specs:** NC7000 from Nanocyl (Sambreville, Belgium). Average diameter 9.5 nm, average length 1.5 µm, purity >90%.

**PLA:** Industrial grade PLA 3052D, Plastika Kritis S.A. (Greece). Supplier tensile strength 62 MPa (ASTM D638). Measured FDM printed 0 wt% control = 45.7 MPa (lower due to 45° off-axis printing orientation).

**Matrix tensile strength used:** 45.7 MPa (measured 0 wt% FDM specimen, not supplier spec).

**Key finding:** ~34% tensile increase at 1.0 wt% MWCNT; plateau at 2.5 and 5.0 wt% — percolation threshold reached at 1.0 wt%.

**Quality gate: PASS** — FDM + ASTM D638 + bar chart with numeric labels readable directly. PLA-filtered relevance: HIGH.

---

## Paper 7: Velu et al. (2025) — Manufacturing Letters — EXTRACTED ✓ PARTIAL

**Citation:** Anand Sankar M, Velu R, Anand Kumar S, Thomas S. Development and characterisation of PLA-MWCNT filaments for multifunctional hybrid composites reinforced with continuous carbon fibre via 3D printing. *Manufacturing Letters.* 2025;44:1139–1148.

**DOI:** 10.1016/j.mfglet.2025.07.004
**Output file:** `per_paper/velu_2025_manufacturing-letters.csv`
**Rows extracted:** 5 (0, 0.5, 2, 4, 6 wt% MWCNT)

### Extraction decisions

**Tensile data source:** Figure 3(b) bar chart — GRAPHICAL only; numeric labels NOT directly readable. Values estimated from bar heights against y-axis (0–50 MPa scale). All tensile values flagged [GRAPHICAL]. Text confirms "nearly 35% increase at 2 wt%" — used to calibrate estimate: neat PLA ~35 MPa, 2 wt% ~47 MPa (35 × 1.35 = 47.25). Other loading estimates: 0.5 wt% ~45 MPa, 4 wt% ~43 MPa, 6 wt% ~44 MPa.

**CRITICAL — Processing route:** Specimens are EXTRUDED FILAMENTS, NOT FDM-printed parts. Processing: melt mixing (HAAKE Rheomix PolyLab QC Lab Mixer, 165°C) → Wellzoom single-screw extruder (10 rpm, 190°C head) → 1.75 ± 0.1 mm diameter filament. Filaments were later used for 3D printing CFRP composites, but the tensile specimens are the filaments themselves.

**Testing standard: NOT ASTM D638.** Equipment: Instron 3344 dynamometer, 2 kN load cell, 1 mm/min crosshead speed. Specimen geometry: trapezoidal dog-bone PLA tabs (non-standard), filament gauge length 80 ± 1 mm. n=3 per material.

**MWCNT specs:** Outer diameter >50 nm, length 10–30 µm, purity 95 wt%. Sisco Research Laboratories India, commercial code 96390.

**PLA:** From Nano Research Lab India, MFR 2–10 g/10 min (210°C, 2.16 kg, ASTM D1238).

**Young's modulus:** Estimated from Figure 3(c) bar chart (GRAPHICAL). Values unusual — neat PLA ~450 MPa (0.45 GPa) is low, rising sharply to ~2.6 GPa at 2 wt%. Likely reflects filament axial stiffness measurement (not bulk modulus) from a non-standard test configuration.

**Quality gate: PARTIAL** — graphical extraction (no numeric labels in bar charts), non-standard testing equipment, non-ASTM specimen geometry, specimens are filaments not FDM-printed parts. PLA-filtered relevance: MEDIUM (PLA + MWCNT material system matches; processing route does NOT pass FDM filter). Do NOT include in HIGH-relevance FDM-only subset.

---

## Quality gate assessment (after 7 papers)

| Paper | Quality Gate | PLA-Filtered Relevance | Rows | Notes |
|---|---|---|---|---|
| Mohammed et al. (2025) | PASS | HIGH | 8 | FDM + ASTM D638 + numeric table |
| Luo et al. (2018) | PARTIAL | HIGH | 5 | FDM + numeric table, testing standard unspecified |
| Hardani et al. (2024) | PASS | HIGH | 15 | FFF + ASTM D638 + numeric table (RSM design) |
| da Silva et al. (2025) | PARTIAL | MEDIUM | 5 | Injection molding (not FDM) + ASTM D638 + partial text values |
| Seimann et al. [Manufacturing Letters] | FAIL | NONE | 0 | Wrong paper retrieved — Ti6Al4V blisk machining |
| Vidakis et al. (2021) | PASS | HIGH | 6 | FFF + ASTM D638 + bar chart numeric labels readable |
| Velu et al. (2025) | PARTIAL | MEDIUM | 5 | Filament specimens (not FDM) + non-standard test equipment + graphical values |

**Total rows extracted across all papers:** 44 (8 + 5 + 15 + 5 + 6 + 5)

**HIGH-relevance rows (PLA + FDM/FFF, PASS quality gate):**
- Mohammed 2025: 8 rows (4 wt% × 2 print orientations; 4 rows if restricted to 0° only)
- Luo 2018: 5 rows (0/1/3/5/7 wt%)
- Hardani 2024: 15 rows (RSM design — effectively 3 wt% levels × multiple print conditions)
- Vidakis 2021: 6 rows (0/0.1/0.5/1.0/2.5/5.0 wt%)
- **Total HIGH-relevance rows: 34** (or 30 if Mohammed restricted to 0° orientation)

**MEDIUM-relevance rows (PLA, non-FDM or PARTIAL):**
- da Silva 2025: 5 rows (injection molding)
- Velu 2025: 5 rows (filament extrusion, non-standard test)

**Decision recommendation for Phase B expansion:**
Pipeline validated across 6 extractable papers. 34 HIGH-relevance rows from 4 papers. Vidakis 2021 is a clean PASS addition: FFF + ASTM D638 + bar chart numeric labels readable directly. Velu 2025 is useful as a material-system reference but must not be pooled with FDM specimens. Hold Phase B expansion (18 remaining OA FLAG papers) until April 30 submission is locked.

**Known schema issues to carry forward:**
1. Testing standard unspecified: Luo 2018. If other papers also omit this, it is a common enough gap to accept as "unspecified" in the dataset.
2. CNT type ambiguity: Hardani 2024 uses "CNT" without "MWCNT" label. Document and flag; do not silently assign MWCNT.
3. Print orientation: Mohammed 2025 (0° vs 90°). Check whether any new papers also test multiple orientations.
4. RSM multi-condition rows: Hardani 2024 produces 15 rows at varying print speed and nozzle temperature. These cannot be directly pooled with single-condition papers without a processing-condition covariate. Flag for adviser decision.
5. Filament vs. FDM specimen split: Velu 2025 adds a new category (filament mechanical testing). Keep this subset separate; do not pool with FDM printed specimen rows in the filtered case-study dataset.
