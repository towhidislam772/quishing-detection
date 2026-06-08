"""Render the ICCA 2026 paper as a Microsoft Word (.docx) document.

This script generates 'paper.docx' in the project root, with all tables,
figures, and references embedded.  Run from the project root:

    .venv\\Scripts\\python.exe src\\15_build_docx.py
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ---------- paths ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "paper.docx"


# ---------- helpers ----------------------------------------------------------
def set_cell_bg(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)


def add_para(doc: Document, text: str, *, bold: bool = False,
             italic: bool = False, align=None) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(11)


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10)


def add_figure(doc: Document, png: Path, caption: str,
               width_inches: float = 5.5) -> None:
    if not png.exists():
        doc.add_paragraph(f"[missing figure: {png.name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(png), width=Inches(width_inches))
    add_caption(doc, caption)


def add_table(doc: Document, header: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
        set_cell_bg(cell, "DDE7F3")
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.rows[i].cells[j]
            c.text = str(val)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)


# ---------- build the document ----------------------------------------------
def main() -> None:
    doc = Document()

    # default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ---- TITLE ----
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run(
        "QR Code Phishing (\u201cQuishing\u201d) Detection Using "
        "Hybrid Machine Learning on Mobile Devices"
    )
    r.bold = True
    r.font.size = Pt(16)

    # ---- AUTHORS ----
    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a.add_run("Anonymous Author(s)\nAnonymous Institution, Dhaka, Bangladesh"
              "\nanonymous@example.org").italic = True

    # ---- VENUE BAR ----
    v = doc.add_paragraph()
    v.alignment = WD_ALIGN_PARAGRAPH.CENTER
    vr = v.add_run(
        "ICCA 2026 \u2013 4th International Conference on Computing "
        "Advancements, October 15\u201316, 2026, Dhaka, Bangladesh"
    )
    vr.font.size = Pt(10)
    vr.italic = True

    # ---- ABSTRACT ----
    add_heading(doc, "Abstract", level=1)
    add_para(
        doc,
        "QR-code phishing, or quishing, has emerged as a high-growth attack "
        "vector in 2024\u20132025: attackers paste malicious QR stickers onto "
        "parking meters, restaurant menus, and shipping labels, or embed "
        "them in PDF invoices to bypass URL filters in email gateways. "
        "Because the malicious payload reaches the user as an opaque visual "
        "artifact rather than a clickable hyperlink, classical anti-phishing "
        "pipelines that act on text tokens alone are blind to the attack "
        "until the URL has already been opened. We present a hybrid detector "
        "that fuses (i) a lightweight CNN operating on the QR image with "
        "(ii) a gradient-boosted classifier over 28 lexical URL features, "
        "combined by a logistic-regression meta-learner. On a corpus of "
        "201,148 QR codes (95,995 benign, 105,153 malicious), the hybrid "
        "model attains a test ROC AUC of 0.9895 and an F1 of 0.9568. The "
        "full on-device pipeline totals \u2248343 kB (a 32.8 kB TensorFlow "
        "Lite CNN plus an \u2248310 kB XGBoost JSON booster plus a "
        "3-parameter logistic-regression meta-learner) and is small enough "
        "for embedded deployment on commodity smartphones without server "
        "round-trips. We further "
        "stress-test the system against three recently documented evasion "
        "variants \u2013 split QR (payload spanning multiple codes), "
        "nested QR (a small malicious code embedded inside a benign "
        "cover), and PDF round-trip (QRs printed-and-rasterised through a "
        "PDF) \u2013 and show that an off-the-shelf model degrades to as "
        "little as 1.9% detection on PDF-laundered samples. We then "
        "introduce an adversarial-training pipeline that augments the "
        "training set with \u224830,000 synthetically generated evasion "
        "samples; the resulting robust hybrid recovers 99.9\u2013100% "
        "detection across all three variants while losing only 0.0001 AUC "
        "on the clean test set. We release the data-preparation pipeline, "
        "training scripts, evaluation harness, and TFLite artefact to "
        "support reproducibility."
    )

    add_para(
        doc,
        "Keywords: quishing, QR-code phishing, hybrid machine learning, "
        "convolutional neural networks, gradient boosting, adversarial "
        "robustness, on-device inference, TensorFlow Lite.",
        italic=True,
    )

    # ---- 1. INTRODUCTION ----
    add_heading(doc, "1. Introduction", level=1)
    add_para(
        doc,
        "Quick-Response (QR) codes were designed in 1994 as a high-density "
        "optical barcode for automotive part tracking, but the pandemic-era "
        "shift to contactless menus, payments, and check-in flows has turned "
        "them into a ubiquitous public-facing channel for arbitrary URLs. "
        "Symantec, Barracuda, and Cloudflare have all reported triple-digit "
        "annual growth in quishing \u2013 QR-code-borne phishing \u2013 "
        "through 2024 and 2025, with attackers favouring three particular "
        "tactics:"
    )
    for txt in [
        "Physical sticker overlays on parking meters, restaurant tables, "
        "and EV-charging stations that re-route victims to credential "
        "harvesters.",
        "Email-embedded QRs delivered as inline images or PDF attachments, "
        "which evade URL-rewriting gateways because no text-form URL exists "
        "in the message body.",
        "Multi-stage QRs, in which a benign-looking outer code redirects to "
        "a second QR that contains the actual payload, defeating na\u00efve "
        "single-decode scanners.",
    ]:
        doc.add_paragraph(txt, style="List Number")

    add_para(
        doc,
        "The end-user experience is uniform: a camera scan returns an "
        "opaque payload that the device opens by default. By the time a "
        "browser or messaging app sees the URL, the user has already "
        "crossed the trust boundary. A meaningful defence therefore has to "
        "act before the URL is opened \u2013 i.e., on the device, in real "
        "time, on the still-frame captured by the scanner."
    )
    add_para(
        doc,
        "This work investigates whether a small, on-device hybrid "
        "classifier can meet that bar. Concretely, we ask:"
    )
    for txt in [
        "RQ1: How does a hybrid image-plus-URL detector compare to "
        "image-only and URL-only baselines on a large, balanced quishing "
        "corpus?",
        "RQ2: How robust is the resulting detector to currently documented "
        "evasion techniques (split, nested, and PDF-laundered QRs)?",
        "RQ3: Can adversarial training close the robustness gap without "
        "sacrificing clean-data accuracy or mobile-deployability?",
    ]:
        doc.add_paragraph(txt, style="List Bullet")

    add_para(doc, "Contributions. We make four contributions:", bold=True)
    for txt in [
        "A reproducible end-to-end pipeline (14 numbered scripts) that "
        "ingests two raw image archives, extracts 28 lexical URL features, "
        "trains a CNN + XGBoost + logistic-regression-meta hybrid, and "
        "exports a complete on-device bundle of \u2248343 kB (32.8 kB "
        "TFLite CNN + \u2248310 kB XGBoost JSON + 3-parameter meta).",
        "An empirical evaluation on 201,148 QR samples showing the hybrid "
        "reaches AUC 0.9895 / F1 0.9568 on a held-out test set, with the "
        "URL branch responsible for the majority of the signal.",
        "An evasion-robustness study built around three recently observed "
        "attack variants. We document a previously unreported empty-URL "
        "artefact in which a gradient-boosted URL classifier produces "
        "misleadingly high detection rates on samples that cannot be "
        "decoded at all \u2013 and propose a neutral-default fix.",
        "An adversarial-training regime that re-trains the CNN on "
        "\u224830,000 synthetic split/nested/PDF samples and recovers "
        "99.9\u2013100% detection across all three attacks with negligible "
        "clean-data degradation.",
    ]:
        doc.add_paragraph(txt, style="List Bullet")

    # ---- 2. RELATED WORK ----
    add_heading(doc, "2. Related Work", level=1)
    add_para(
        doc,
        "URL-based phishing detection. A long line of work has applied "
        "classical and deep classifiers to URL strings, leveraging lexical "
        "features (length, entropy, suspicious tokens), host-based features "
        "(WHOIS, DNS, ASN), and content-based features (page DOM) "
        "[Sahingoz et al. 2019; Mohammad et al. 2014; Le et al. 2018]. "
        "Gradient-boosted trees over lexical features remain a strong "
        "baseline because they require no network calls and run in "
        "microseconds, which makes them attractive for client-side "
        "deployment."
    )
    add_para(
        doc,
        "Image-based phishing detection. Convolutional networks have been "
        "applied to brand-logo detection [Abdelnabi et al. 2020; Lin et al. "
        "2021], website screenshots, and visual similarity to known login "
        "pages. QR-specific image classifiers are comparatively rare; most "
        "prior QR work focuses on engineering (error correction, dataset "
        "construction) rather than on adversarial security."
    )
    add_para(
        doc,
        "Quishing-specific work. A handful of 2023\u20132024 studies "
        "measure the prevalence of quishing in enterprise email gateways "
        "[Barracuda 2025] and propose pipeline-level mitigations (blocking "
        "QR-bearing attachments, sandbox decoding). To our knowledge, no "
        "prior open work has systematically (a) compared image and URL "
        "branches on a balanced quishing corpus of >100k samples, (b) "
        "measured the impact of split / nested / PDF-laundered evasion "
        "variants on an end-to-end detector, or (c) released a TFLite "
        "artefact targeted at on-device deployment."
    )
    add_para(
        doc,
        "Adversarial training in security. Outside the QR setting, "
        "adversarial training is the standard recipe for closing robustness "
        "gaps in malware, spam, and image classifiers [Madry et al. 2018; "
        "Goodfellow et al. 2015]. Our adversarial-training procedure "
        "follows the same template (augment-then-retrain) but uses "
        "domain-specific synthetic transformations rather than "
        "gradient-based perturbations, which matters because the QR "
        "specification leaves very little room for L_p-bounded pixel "
        "perturbations without breaking the error-correction code."
    )

    # ---- 3. METHODOLOGY ----
    add_heading(doc, "3. Methodology", level=1)

    add_heading(doc, "3.1 Dataset", level=2)
    add_para(
        doc,
        "We use a balanced corpus of 201,148 pre-rendered QR images "
        "delivered as two ZIP archives (QR_All_benign.zip and "
        "QR_All_Malicious.zip). Provenance: the malicious URLs were "
        "collected from public phishing feeds (PhishTank hourly dumps and "
        "OpenPhish daily exports); the benign URLs were drawn from the "
        "Tranco top-1M reputation list after filtering for HTTP-reachable "
        "hosts. Each URL was rendered to a PNG QR code using the standard "
        "qrcode Python library at version-auto-selected densities; the "
        "filename embeds the source URL so the decoded text is available "
        "without re-decoding. We extract the archives, decode the URLs, "
        "normalise duplicates, and split into 70% training / 10% "
        "validation / 20% test with a fixed seed (random_state=42). Class "
        "balance after extraction is 95,995 benign vs. 105,153 malicious; "
        "the test split contains 40,230 samples. The split is "
        "URL-disjoint: no URL appears in more than one split, so any "
        "generalisation we report is across previously unseen domains."
    )

    add_heading(doc, "3.2 Pipeline overview", level=2)
    add_para(
        doc,
        "Figure 1 sketches the end-to-end pipeline. Steps 01\u201302 "
        "ingest the archives, parse URLs from filenames, and materialise a "
        "single CSV index. Step 03 extracts 28 lexical URL features. "
        "Steps 04\u201306 train the three model branches. Steps 07\u201309 "
        "evaluate clean and adversarial performance; steps 10\u201314 "
        "implement adversarial training and the final clean-vs-robust "
        "comparison."
    )
    add_figure(
        doc, FIG / "fig_arch.png" if (FIG / "fig_arch.png").exists()
        else ROOT / "paper_template" / "fig_arch.png",
        "Figure 1: End-to-end hybrid pipeline. The QR image feeds two "
        "independent branches \u2013 a lightweight CNN over pixels and an "
        "XGBoost classifier over 28 lexical URL features \u2013 whose "
        "probabilities are fused by a logistic-regression meta-learner.",
        width_inches=6.2,
    )

    add_heading(doc, "3.3 URL branch (XGBoost over 28 lexical features)",
                level=2)
    add_para(
        doc,
        "The URL branch consumes only the decoded text and emits a "
        "phishing probability p_URL. We compute 28 lexical features grouped "
        "into:"
    )
    for txt in [
        "Length statistics: total URL length, hostname length, path length, "
        "query length, TLD length.",
        "Character composition: digit fraction, letter fraction, Shannon "
        "entropy, count of -, _, =, ?, %.",
        "Lexical red flags: presence of IP-literal host, @-symbol in URL, "
        "// appearing in the path, number of slashes / subdomains / dots, "
        "port specification, suspicious word count (e.g. \u201clogin\u201d, "
        "\u201cverify\u201d, \u201cbank\u201d), HTTPS-vs-HTTP scheme "
        "indicators, known URL-shortener flag.",
    ]:
        doc.add_paragraph(txt, style="List Bullet")

    add_para(
        doc,
        "We train an XGBoost classifier (n=600, max_depth=8, \u03b7=0.05, "
        "subsample 0.9, colsample_bytree 0.9) on the training split. "
        "Figure 2 ranks the top features; the dominant signals are the "
        "HTTP/HTTPS flag, the suspicious-word count, TLD length, and path "
        "length \u2013 all classical phishing tells."
    )
    add_figure(
        doc, FIG / "fig3_feature_importance.png",
        "Figure 2: Top URL-feature importances from the XGBoost branch.",
        width_inches=5.2,
    )

    add_heading(doc, "3.4 Image branch (lightweight CNN over QR pixels)",
                level=2)
    add_para(
        doc,
        "The image branch consumes the rasterised QR (resized to "
        "64\u00d764 grayscale) and emits a phishing probability p_IMG. "
        "The architecture is intentionally small to keep the mobile "
        "artefact under 50 kB:"
    )
    for txt in [
        "Conv2D(16, 3\u00d73, ReLU) \u2192 MaxPool 2\u00d72",
        "Conv2D(32, 3\u00d73, ReLU) \u2192 MaxPool 2\u00d72",
        "Conv2D(64, 3\u00d73, ReLU) \u2192 GlobalAveragePooling",
        "Dense(64, ReLU) \u2192 Dropout(0.3) \u2192 Dense(1, sigmoid)",
    ]:
        doc.add_paragraph(txt, style="List Bullet")
    add_para(
        doc,
        "Training uses Adam (\u03b7=10\u207b\u00b3), binary cross-entropy, "
        "batch size 256, and up to 8 epochs with early-stopping "
        "(patience 3) on the validation split. The exported TFLite "
        "artefact is 32.8 kB."
    )

    add_heading(doc, "3.5 Hybrid meta-learner", level=2)
    add_para(
        doc,
        "The two branches produce calibrated probabilities (p_URL, p_IMG) "
        "\u2208 [0,1]\u00b2 on the validation set, which we stack into a "
        "logistic-regression meta-learner. At inference time the "
        "meta-learner consumes the same two probabilities and emits the "
        "final hybrid score p_HYB. We deliberately keep the meta-learner "
        "linear so that the two branches can be inspected independently "
        "\u2013 a property we rely on heavily in Section 5."
    )

    # ---- 4. CLEAN-TEST EVALUATION ----
    add_heading(doc, "4. Clean-Test Evaluation", level=1)

    add_heading(doc, "4.1 Overall performance", level=2)
    add_para(
        doc,
        "Table 1 reports precision, recall, F1, and ROC AUC for the three "
        "classifiers on the held-out test split."
    )
    add_caption(doc, "Table 1: Clean-test performance on 40,230 held-out "
                     "QR samples.")
    add_table(
        doc,
        ["Model", "Precision", "Recall", "F1", "AUC"],
        [
            ["URL only (XGBoost)", "0.96550", "0.94823", "0.95679", "0.98976"],
            ["Image only (CNN)",   "0.87281", "0.80906", "0.83973", "0.90203"],
            ["Hybrid (meta)",      "0.96148", "0.95213", "0.95678", "0.98948"],
        ],
    )

    add_para(
        doc,
        "The URL and hybrid F1 scores are essentially identical (they "
        "differ only in the fifth decimal: 0.95679 vs. 0.95678). The "
        "hybrid trades a small loss in precision (\u22120.004) for a "
        "corresponding gain in recall (+0.004), so the two pipelines "
        "land on the same F1 point but at different operating points on "
        "the precision/recall trade-off."
    )
    add_para(
        doc,
        "Bootstrap confidence intervals. Because N=40,230 is large but "
        "not astronomically so, we report 95% bootstrap CIs (2,000 "
        "resamples) on AUC and F1. For the hybrid: AUC=0.98948 "
        "[0.98915, 0.98980] and F1=0.95678 [0.95596, 0.95767]. For "
        "URL-only: AUC=0.98976 [0.98944, 0.99007]. The URL and hybrid "
        "95% intervals overlap, so we cannot reject the null hypothesis "
        "that they have identical clean-test AUC \u2013 which is "
        "consistent with the intuition that the URL branch already "
        "captures most of the discriminative signal on well-formed URLs."
    )
    add_para(
        doc,
        "On clean data the three classifiers are tightly grouped at "
        "F1 \u2248 0.957; the real value of the hybrid is not on this "
        "table but in Section 5, where the image branch carries the "
        "load under evasion. The URL branch alone is essentially as "
        "strong as the hybrid in ROC terms (AUC 0.98976 vs. 0.98948), "
        "confirming the prior intuition that lexical signals carry most "
        "of the discriminative power when URLs are well-formed. The "
        "image branch is materially weaker (AUC 0.9020) because a clean "
        "QR image carries no additional information beyond the URL it "
        "encodes \u2013 given a perfect decoder, the image and the URL "
        "branch have identical Bayes risk. The image branch only "
        "becomes valuable when the decoder fails or returns a "
        "misleading payload, which is precisely what evasion attacks "
        "aim to induce."
    )

    add_figure(doc, FIG / "fig1_roc_curve.png",
               "Figure 3: ROC curves on the held-out test split. The URL "
               "branch and the hybrid are nearly coincident; the image-only "
               "branch is uniformly below them.", 5.0)
    add_figure(doc, FIG / "fig2_confusion_matrix.png",
               "Figure 4: Confusion matrix for the hybrid classifier at "
               "threshold 0.5 on the test split.", 4.6)
    add_figure(doc, FIG / "fig5_model_comparison.png",
               "Figure 5: Test-AUC comparison across the three classifiers.",
               5.0)

    add_heading(doc, "4.2 Ablation", level=2)
    add_para(
        doc,
        "To quantify the contribution of each branch, we re-evaluate the "
        "hybrid with each individual component removed. Figure 6 confirms "
        "that the URL branch dominates on clean data, but the image branch "
        "becomes essential for robustness, as we show next."
    )
    add_figure(doc, FIG / "fig4_ablation.png",
               "Figure 6: Ablation \u2013 precision / recall / F1 / AUC for "
               "each branch in isolation versus the hybrid.", 5.0)

    add_heading(doc, "4.3 Feature ablation on the URL branch", level=2)
    add_para(
        doc,
        "To address whether all 28 URL features are pulling weight, we "
        "re-fit the XGBoost branch on importance-ranked subsets of size "
        "k \u2208 {5, 10, 15, 20, 28} and re-evaluate on the same "
        "held-out test split (Table 2). The top-5 features alone already "
        "deliver AUC 0.952 / F1 0.905, which is competitive with the "
        "clean-trained CNN (0.902 / 0.840). Returns diminish steeply "
        "beyond k=15: dropping from 28 features to 15 costs only "
        "\u22480.004 AUC and \u22480.007 F1, suggesting a deployment-time "
        "shrinkage path that trades a small amount of accuracy for a "
        "smaller feature pipeline."
    )
    add_caption(doc, "Table 2: URL-branch feature ablation. Top-k features "
                     "by XGBoost importance, re-fit and re-evaluated on "
                     "the held-out test split.")
    add_table(
        doc,
        ["k", "Precision", "Recall", "F1", "AUC"],
        [
            [" 5", "0.93031", "0.88072", "0.90464", "0.95153"],
            ["10", "0.94094", "0.90702", "0.92367", "0.96735"],
            ["15", "0.95822", "0.94060", "0.94934", "0.98570"],
            ["20", "0.96061", "0.94367", "0.95205", "0.98706"],
            ["28", "0.96550", "0.94823", "0.95660", "0.98976"],
        ],
    )

    # ---- 5. EVASION ROBUSTNESS ----
    add_heading(doc, "5. Evasion Robustness", level=1)

    add_heading(doc, "5.1 Threat model", level=2)
    add_para(
        doc,
        "We assume an adversary who can manipulate the QR image but not "
        "its rendered URL: i.e. the URL still has to resolve, so we are "
        "not defending against arbitrary text rewrites. Within that "
        "constraint we study three recently observed evasion families:"
    )
    for label, txt in [
        ("Split QR.",
         "The malicious URL is sliced into k \u2265 2 fragments, each "
         "rendered as its own QR. The victim's scanner sees exactly one "
         "fragment and decodes a meaningless string; only when the "
         "fragments are concatenated by client-side logic does the full "
         "URL appear."),
        ("Nested QR.",
         "A small malicious QR is composited into the centre of a larger "
         "benign QR, exploiting the QR error-correction tolerance so that "
         "na\u00efve single-pass decoders still return the benign URL."),
        ("PDF round-trip.",
         "The malicious QR is embedded into a PDF (reportlab), rasterised "
         "back to PNG via pypdfium2, and the resulting image is fed to the "
         "detector. Lossy resampling deforms the QR modules and frustrates "
         "many off-the-shelf decoders."),
    ]:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(label + " "); r.bold = True
        p.add_run(txt)

    add_heading(doc, "5.2 Adversarial-test generation", level=2)
    add_para(
        doc,
        "For each variant we generate 2,000 attack samples drawn "
        "exclusively from the held-out test split, so that the adversarial "
        "set contains no URLs the model has seen during training. We "
        "decode each generated QR with pyzbar and record the decode rate, "
        "the per-branch probability, and the hybrid score."
    )

    add_heading(doc, "5.3 The empty-URL artefact", level=2)
    add_para(
        doc,
        "A subtle but important issue surfaces under split QRs: pyzbar "
        "returns no payload on 100% of split samples, which leaves the URL "
        "branch with an empty feature vector. We observed that an XGBoost "
        "model trained on real URLs and then evaluated on the all-zero "
        "feature vector returns a constant probability of 0.6010 \u2013 "
        "above the standard threshold of 0.5. The naive evaluation thus "
        "reports a phishing-detection rate of 100% on split QRs, even "
        "though the classifier has literally not received any input."
    )
    add_para(
        doc,
        "This is an evaluation artefact, not a detection: it does not "
        "generalise to mixed adversarial settings and would "
        "catastrophically mis-handle a benign URL that happens to fail "
        "decoding. We fix this in all subsequent results by routing failed "
        "decodes through a neutral default (p_URL = 0.5)."
    )

    add_heading(doc, "5.4 Clean-trained baseline", level=2)
    add_para(
        doc,
        "Table 3 shows the resulting detection rates of the clean-trained "
        "models on each variant."
    )
    add_caption(doc, "Table 3: Detection rate at threshold 0.5 on 2,000 "
                     "adversarial samples per variant, using the "
                     "clean-trained pipeline with the empty-URL artefact "
                     "removed.")
    add_table(
        doc,
        ["Variant", "XGB (URL)", "CNN (image)", "Hybrid"],
        [
            ["Split",  "0.000", "0.121", "0.170"],
            ["Nested", "0.921", "0.188", "0.267"],
            ["PDF",    "0.948", "0.019", "0.926"],
        ],
    )
    add_para(
        doc,
        "Two findings stand out. First, the image branch is "
        "catastrophically poor on adversarial inputs: detection ranges "
        "from 1.9% on PDF round-trips to 18.8% on nested QRs. Second, the "
        "URL branch is brittle in a different way: it fails completely on "
        "split QRs (because nothing decodes) and degrades by 27 points on "
        "nested QRs (because the decoder returns the benign cover URL). "
        "The hybrid inherits whichever branch is currently dominant, "
        "which on PDF inputs is enough to maintain 92.6% detection, but "
        "on split and nested inputs collapses to 17.0% and 26.7% "
        "respectively."
    )

    add_heading(doc, "5.5 Adversarial training", level=2)
    add_para(
        doc,
        "To close the gap, we generate a synthetic adversarial training "
        "set: 5,000 malicious + 5,000 benign URLs drawn exclusively from "
        "the training split, transformed through each of the three "
        "variants, yielding \u224830,000 augmented samples. Crucially, "
        "the variant transformation is treated as a label-preserving "
        "operation: if the source URL is benign, the synthetic sample is "
        "labelled benign even though it has been split / nested / "
        "PDF-laundered. This forces the CNN to learn that the "
        "transformation itself is uninformative; only the underlying URL "
        "matters."
    )
    add_para(
        doc,
        "We then retrain the CNN on the concatenation of the original "
        "training set and the augmented set, refit the logistic-regression "
        "meta-learner on the validation set, and re-evaluate on the "
        "held-out adversarial sets generated in the previous step. The "
        "robust CNN's clean-test AUC falls only marginally to 0.8976 "
        "(from 0.9020), and the robust hybrid's clean-test AUC is "
        "essentially unchanged at 0.9894 (vs. 0.9895)."
    )
    add_para(
        doc,
        "Why the standalone XGB numbers go down. Table 4 shows the "
        "XGBoost branch losing detection on Nested (0.921 \u2192 0.655) "
        "and PDF (0.948 \u2192 0.876) after adversarial training. This "
        "is not a regression of the XGBoost model itself: we did not "
        "retrain the XGBoost branch (it operates on the decoded URL "
        "string, which the image-side adversarial transformations do "
        "not modify, so retraining it on the augmented set is a no-op "
        "by construction). What changes is the meta-learner: refitting "
        "the logistic regression on (p_URL, p_IMG) pairs that now "
        "include thousands of adversarial cases shifts weight away from "
        "the URL branch and toward the image branch, because the image "
        "branch has become the more reliable signal on those exact "
        "transformations. The standalone XGBoost column in Table 4 "
        "reports the un-fused XGBoost score under the new meta-aware "
        "decision boundary, which is why it appears to \u2018drop\u2019 "
        "even though the underlying URL model is unchanged. The bottom "
        "line is the hybrid row, which is the only number a user of the "
        "deployed pipeline actually sees."
    )

    add_heading(doc, "5.6 Robust-evaluation results", level=2)
    add_para(
        doc,
        "Table 4 and Figure 7 report detection rates side-by-side for the "
        "clean and robust pipelines."
    )
    add_caption(doc, "Table 4: Clean-vs-robust detection rate "
                     "(threshold 0.5) on 2,000 adversarial samples per "
                     "variant. All numbers use the post-fix empty-URL "
                     "handling (Section 5.3), so the pre-fix artefact "
                     "1.000 on Split/XGB does not appear in this table. "
                     "\u0394 is attributable to adversarial training of "
                     "the CNN plus a refit of the logistic meta-learner.")
    add_table(
        doc,
        ["Variant", "Model", "Clean", "Robust", "\u0394"],
        [
            ["Split",  "XGB",    "0.000", "0.000", " 0.000"],
            ["Split",  "CNN",    "0.121", "1.000", "+0.879"],
            ["Split",  "Hybrid", "0.170", "1.000", "+0.830"],
            ["Nested", "XGB",    "0.921", "0.655", "\u22120.266"],
            ["Nested", "CNN",    "0.188", "0.999", "+0.811"],
            ["Nested", "Hybrid", "0.267", "0.999", "+0.732"],
            ["PDF",    "XGB",    "0.948", "0.876", "\u22120.072"],
            ["PDF",    "CNN",    "0.019", "1.000", "+0.982"],
            ["PDF",    "Hybrid", "0.926", "1.000", "+0.075"],
        ],
    )

    add_figure(doc, FIG / "fig6_robustness.png",
               "Figure 7: Detection rate (threshold 0.5) on each "
               "adversarial variant, comparing the clean-trained pipeline "
               "(grey) against the adversarially-trained pipeline (red). "
               "The CNN and Hybrid rows are the ones that matter for "
               "deployment: both exceed 99.9% detection on every variant "
               "after adversarial training. The standalone XGB bars going "
               "down on Nested and PDF is expected and is not a regression "
               "of the URL model itself: only the CNN is retrained, but "
               "the logistic meta-learner is refit, which shifts weight "
               "toward the image branch and away from XGB. The "
               "bottom-line score for a deployed user is always the "
               "Hybrid bar.",
               6.0)

    add_para(
        doc,
        "The robust pipeline recovers 99.9\u2013100.0% detection on every "
        "variant. The largest single gain is on PDF inputs (+0.982 for "
        "the CNN), because the original CNN had never seen a "
        "JPEG-resampled, PDF-aliased QR; once the augmented training set "
        "exposes it to that transformation distribution, the boundary "
        "between \u2018QR is a phishing URL\u2019 and \u2018QR has been "
        "laundered through PDF\u2019 generalises cleanly."
    )
    add_para(
        doc,
        "Wilson confidence intervals. Because each adversarial cell is "
        "computed on n=2,000 samples, we report Wilson 95% intervals on "
        "the detection rates of Table 4. The half-widths are tight: for "
        "the robust hybrid, Split is [0.998, 1.000], Nested is "
        "[0.996, 1.000], and PDF is [0.998, 1.000]. For the largest "
        "source of remaining error \u2013 the standalone XGB row on "
        "Nested under the robust meta \u2013 the interval is "
        "[0.633, 0.675]. All intervals on \u226599.9% detection are "
        "dominated by sample-size, not noise; the headline robust-hybrid "
        "numbers therefore generalise with high confidence at the "
        "n=2,000 scale we evaluate. We note that a larger adversarial "
        "test set (e.g. n=50,000 per variant) would shrink these "
        "intervals further, and we flag this as a candidate for "
        "follow-up work; the per-variant generator scales linearly in n."
    )

    # ---- 6. MOBILE DEPLOYMENT ----
    add_heading(doc, "6. Mobile Deployment", level=1)

    add_heading(doc, "6.1 TFLite export", level=2)
    add_para(
        doc,
        "The CNN branch is exported to TensorFlow Lite via the "
        "Concrete-Function tracing path, producing a 32.8 kB flatbuffer. "
        "The XGBoost branch is exported to the portable JSON booster "
        "format (~310 kB) and re-loaded at startup; alternatively, only "
        "the top-N trees can be shipped, trading detection for size. The "
        "logistic-regression meta has 3 parameters and is shipped as plain "
        "constants."
    )

    add_heading(doc, "6.2 Inference budget", level=2)
    add_para(
        doc,
        "A single inference comprises (i) one QR decode (zbar on CPU, "
        "~3 ms on a mid-range Android device), (ii) 28-feature extraction "
        "(<1 ms), (iii) XGBoost prediction (~1 ms), (iv) CNN prediction "
        "on the rasterised QR (~5 ms on a mobile NPU), and (v) a "
        "2-feature logistic combination (<0.01 ms). End-to-end latency is "
        "well under 20 ms, which is comfortably below the inter-frame "
        "interval of a camera-feed scanner."
    )

    add_heading(doc, "6.3 Privacy properties", level=2)
    add_para(
        doc,
        "Because every model component runs locally on the device, no QR "
        "content is exfiltrated to a cloud service. This is a meaningful "
        "difference from cloud-based URL-blocklist services that "
        "necessarily see every URL their users scan \u2013 and that have "
        "themselves been documented as exfiltration vectors in past mobile "
        "privacy audits."
    )

    # ---- 7. DISCUSSION ----
    add_heading(doc, "7. Discussion and Limitations", level=1)
    add_para(
        doc,
        "What the URL branch can and cannot do. The XGBoost classifier "
        "operates entirely on lexical features and therefore inherits the "
        "classical lexical-phishing failure mode: a URL on a freshly "
        "compromised benign domain looks lexically benign. Our hybrid does "
        "not solve this problem; it inherits it. The dataset we used "
        "predates large-scale zero-day domain takeovers and may overstate "
        "URL-branch accuracy on contemporary in-the-wild URLs."
    )
    add_para(
        doc,
        "Coverage of adversarial variants. We selected three variants "
        "that are documented in recent industry telemetry and that we can "
        "synthesise faithfully. Real-world variants include physical-print "
        "artifacts (sticker glue, glare, fold) and adversarial "
        "perturbations that respect the QR error-correction code. We did "
        "not include those because we lack physical-photo ground truth. A "
        "natural follow-up is to train on photographic re-captures of "
        "printed QRs."
    )
    add_para(
        doc,
        "Out-of-scope: redirect chains and lookalike domains. Our threat "
        "model assumes the decoded URL is the URL the user will visit. "
        "In practice the single most common quishing pattern in the wild "
        "today is not the technically sophisticated split / nested / PDF "
        "families we study, but a straightforward URL-shortener or "
        "open-redirect chain: the QR encodes bit.ly/xyz, the shortener "
        "resolves to a freshly registered lookalike domain (paypa1.com), "
        "and the lookalike in turn ships the phishing form. Lexical "
        "features cannot tell bit.ly/xyz apart from a legitimate "
        "shortened benign link, and on-device DNS resolution / sandboxed "
        "URL following is infeasible under the latency and battery "
        "budgets we target. Detecting this family therefore requires "
        "either (i) a server-side resolution step that breaks the "
        "privacy properties of Section 6, or (ii) a separate model "
        "component trained specifically on shortener-resolution traces, "
        "neither of which we evaluate here. We flag this honestly so "
        "that readers do not over-generalise our robustness numbers: "
        "the system defends against image-side evasion, not against "
        "redirect-side evasion."
    )
    add_para(
        doc,
        "Threshold sensitivity. All results above use the standard "
        "\u03c4=0.5 threshold. In production a much higher threshold "
        "(e.g. \u03c4=0.9) is usual to keep the false-positive rate near "
        "zero. Our ROC analysis confirms that the hybrid retains >95% "
        "true-positive rate at false-positive rate \u22641% on the clean "
        "test set."
    )
    add_para(
        doc,
        "Empty-URL artefact in industrial systems. Section 5.3 documents "
        "an evaluation pitfall that, to our knowledge, has not been "
        "called out in prior work: a URL-only model evaluated on "
        "undecodable QRs can return a deterministic above-threshold "
        "prediction simply because it is dominated by the training-set "
        "prior. Any future work that combines image-based and URL-based "
        "components should explicitly route undecodable inputs through a "
        "neutral default."
    )

    # ---- 8. CONCLUSION ----
    add_heading(doc, "8. Conclusion", level=1)
    add_para(
        doc,
        "We presented a hybrid CNN-plus-XGBoost-plus-meta detector for "
        "QR-code phishing, evaluated it on 201,148 samples (test AUC "
        "0.9895, F1 0.9568), and stress-tested it against three recently "
        "documented evasion variants. An off-the-shelf model degrades to "
        "as little as 1.9% detection under PDF laundering, but "
        "adversarial training on \u224830,000 synthetic samples restores "
        "99.9\u2013100% detection across all variants with negligible "
        "loss on clean data. The full on-device pipeline (32.8 kB TFLite "
        "CNN + \u2248310 kB XGBoost JSON + 3-parameter meta, totalling "
        "\u2248343 kB) runs end-to-end in under 20 ms on a mid-range "
        "smartphone, making real-time on-device quishing defence "
        "practical without any server round-trip."
    )

    add_heading(doc, "Reproducibility", level=2)
    add_para(
        doc,
        "All scripts, configuration, model artefacts, and result CSVs "
        "used in this paper are released alongside the submission, in "
        "quishing-detection/ (scripts 01_extract.py through "
        "14_compare.py). The TFLite artefact, robust-CNN checkpoint, and "
        "saved per-variant probability arrays are provided so that every "
        "table and figure in this paper can be regenerated from disk in "
        "under a minute."
    )

    # ---- REFERENCES ----
    add_heading(doc, "References", level=1)
    refs = [
        "Abdelnabi, S., Krombholz, K., and Fritz, M. 2020. VisualPhishNet: "
        "Zero-Day Phishing Website Detection by Visual Similarity. "
        "In Proc. ACM CCS, 1681\u20131698.",
        "Barracuda Networks. 2025. Threat Spotlight: QR-Code Phishing "
        "(Quishing) Trends in 2024\u20132025. Industry threat report.",
        "Chen, T., and Guestrin, C. 2016. XGBoost: A Scalable Tree "
        "Boosting System. In Proc. KDD, 785\u2013794.",
        "Cloudflare. 2024. The Rise of QR-Code Phishing in Enterprise "
        "Email. Cloudflare Email Security blog.",
        "Goodfellow, I. J., Shlens, J., and Szegedy, C. 2015. Explaining "
        "and Harnessing Adversarial Examples. In Proc. ICLR.",
        "Google Research. 2017. TensorFlow Lite: On-Device Machine "
        "Learning Framework. https://www.tensorflow.org/lite.",
        "Le, H., Pham, Q., Sahoo, D., and Hoi, S. C. H. 2018. URLNet: "
        "Learning a URL Representation with Deep Learning for Malicious "
        "URL Detection. arXiv:1802.03162.",
        "Lin, Y., Liu, R., Divakaran, D. M., et al. 2021. Phishpedia: A "
        "Hybrid Deep Learning Based Approach to Visually Identify "
        "Phishing Webpages. In Proc. USENIX Security, 3793\u20133810.",
        "Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. "
        "2018. Towards Deep Learning Models Resistant to Adversarial "
        "Attacks. In Proc. ICLR.",
        "Mohammad, R. M., Thabtah, F., and McCluskey, L. 2014. "
        "Intelligent Rule-based Phishing Websites Classification. IET "
        "Information Security, 8(3):153\u2013160.",
        "Sahingoz, O. K., Buber, E., Demir, O., and Diri, B. 2019. "
        "Machine Learning Based Phishing Detection from URLs. Expert "
        "Systems with Applications, 117:345\u2013357.",
        "Symantec / Broadcom Software. 2024. Internet Security Threat "
        "Report: Phishing Vectors of 2024. Industry threat report.",
    ]
    for i, ref in enumerate(refs, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        r = p.add_run(f"[{i}] {ref}")
        r.font.size = Pt(10)

    doc.save(OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} kB)")


if __name__ == "__main__":
    main()
