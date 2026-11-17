import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def set_cell_margins(cell, top=40, bottom=40, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_cell_borders(cell, top="none", bottom="none", left="none", right="none", sz="4", color="auto"):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}>'
                          f'<w:top w:val="{top}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
                          f'<w:left w:val="{left}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
                          f'<w:bottom w:val="{bottom}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
                          f'<w:right w:val="{right}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
                          f'</w:tcBorders>')
    tcPr.append(tcBorders)

def make_columns(section, num_cols=2, col_space=360):
    sectPr = section._sectPr
    cols = sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), str(num_cols))
        cols[0].set(qn('w:space'), str(col_space))
    else:
        new_cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="{num_cols}" w:space="{col_space}"/>')
        sectPr.append(new_cols)

doc = docx.Document()

# 1. Base Margins
section1 = doc.sections[0]
section1.top_margin = Inches(0.75)
section1.bottom_margin = Inches(0.75)
section1.left_margin = Inches(0.625)
section1.right_margin = Inches(0.625)

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(10)
font.color.rgb = RGBColor(0, 0, 0)

# Title
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after = Pt(6)
r_title = p_title.add_run("Robustness of Roman Urdu Sentiment Classification to Orthographic Spelling Variation: A Comparative Study of Full Fine-Tuning and Parameter-Efficient Adaptation of Multilingual Transformers")
r_title.font.name = 'Times New Roman'
r_title.font.size = Pt(18)
r_title.font.bold = True

# Sole Author: Danyal Ahmed Malik
p_auth = doc.add_paragraph()
p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_auth.paragraph_format.space_before = Pt(2)
p_auth.paragraph_format.space_after = Pt(8)
r_a1 = p_auth.add_run("Danyal Ahmed Malik\n")
r_a1.bold = True
r_a1.font.size = Pt(11)
r_a2 = p_auth.add_run("Department of Data Science — Graduate Studies\nNational University of Computer & Emerging Sciences (FAST-NUCES)\nIslamabad, Pakistan\n")
r_a2.italic = True
r_a2.font.size = Pt(9.5)
r_a3 = p_auth.add_run("i268030@isb.nu.edu.pk")
r_a3.font.name = "Courier New"
r_a3.font.size = Pt(9)

# Continuous Section Break for 2-column layout
section2 = doc.add_section(WD_SECTION.CONTINUOUS)
section2.top_margin = Inches(0.75)
section2.bottom_margin = Inches(0.75)
section2.left_margin = Inches(0.625)
section2.right_margin = Inches(0.625)
make_columns(section2, 2, 360)

def add_heading(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.upper())
    r.font.name = 'Times New Roman'
    r.font.size = Pt(10)
    r.bold = True
    return p

def add_subheading(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(9.5)
    r.bold = True
    r.italic = True
    return p

def add_p(text, indent=True, bold_prefix="", italic_prefix=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.18)
    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.bold = True
    if italic_prefix:
        r_i = p.add_run(italic_prefix)
        r_i.italic = True
    r_t = p.add_run(text)
    return p

# Abstract Box
p_abs = doc.add_paragraph()
p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_abs.paragraph_format.space_before = Pt(0)
p_abs.paragraph_format.space_after = Pt(4)
p_abs.paragraph_format.line_spacing = 1.05
r_abs_b = p_abs.add_run("Abstract—")
r_abs_b.bold = True
r_abs_b.italic = True
r_abs_t = p_abs.add_run("Roman Urdu has become the dominant vernacular medium for informal digital communication across South Asia. However, because it lacks a standardized orthography, identical phonemes and lexical items manifest in widely divergent spelling forms. While multilingual transformer architectures such as mBERT and XLM-RoBERTa have achieved state-of-the-art results on clean benchmark splits, their robustness under systematic, graded orthographic noise remains largely unexamined. Furthermore, it remains unknown whether Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) confers superior regularized resilience compared to full fine-tuning. In this study, we present a rigorous, end-to-end empirical investigation into Roman Urdu orthographic robustness. Conducting an independent data audit on the 28,090-review RUWV-NSR dataset, we resolve severe label corruption and eliminate 4,632 exact duplicate reviews (16.49%) to establish a strictly leakage-free benchmark. Pairing the corpus with a 5,244-word authentic spelling-variation lexicon (10,869 variant pairs), we design controlled canonical perturbation conditions (Mild, Moderate, Severe) alongside a full 3×3 factorial grid disentangling substitution density from character edit distance. We comprehensively evaluate TF-IDF + Logistic Regression, fully fine-tuned mBERT, fully fine-tuned XLM-RoBERTa, and LoRA-adapted XLM-RoBERTa. Our findings demonstrate that LoRA adaptation exhibits substantially superior orthographic resilience, incurring only a 4.68% relative Macro-F1 degradation under severe noise compared to 7.01% for fully fine-tuned XLM-RoBERTa and 16.42% for the classical baseline. Token fertility profiling reveals that SentencePiece tokenization suffers significantly less fragmentation than WordPiece under variant spellings. Additionally, normalization ablations prove that research-specific lexicon mapping recovers 84.23%–93.79% of spelling-induced performance loss, whereas generic preprocessing yields zero or negative recovery. External validation across 3,000 independent Daraz.pk e-commerce reviews confirms the domain generalizability of these findings.")
r_abs_t.bold = True

p_kw = doc.add_paragraph()
p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_kw.paragraph_format.space_before = Pt(0)
p_kw.paragraph_format.space_after = Pt(6)
r_kw_b = p_kw.add_run("Keywords—")
r_kw_b.bold = True
r_kw_b.italic = True
r_kw_t = p_kw.add_run("Roman Urdu, Sentiment Analysis, Orthographic Robustness, Multilingual Transformers, Low-Rank Adaptation (LoRA), Parameter-Efficient Fine-Tuning (PEFT), Spelling Perturbation.")
r_kw_t.bold = True

# Section I
add_heading("I. Introduction")
add_p("Roman Urdu—the phonological transcription of the Urdu language using the standard 26-letter Latin alphabet—has emerged as the de facto written medium for informal digital communication across Pakistan and the broader South Asian diaspora. It permeates user reviews on e-commerce platforms, customer feedback on social media channels, and mobile instant messaging. Despite its ubiquity, Roman Urdu is strictly an unstandardized vernacular. While classical Nastaliq Urdu employs an alphabet of 36 distinct characters with formal orthographic rules, Roman Urdu lacks any centralized standardizing authority or agreed-upon phonetic mapping. Consequently, native speakers write purely by acoustic approximation.")
add_p("A single lexical item or sentiment-bearing predicate can easily manifest in several divergent spelling variations depending entirely on individual typists, regional phonology, and texting conventions. For instance, the core affirmation accha (\"good\") frequently appears as acha, achha, achaah, or achaaa; similarly, the negation nahi (\"not\") is written interchangeably as nhn, nahin, nhi, or nahe. In the context of automated Natural Language Processing (NLP), and Sentiment Analysis (SA) in particular, this extreme orthographic entropy poses a formidable challenge.")
add_p("In contemporary NLP research, Pre-trained Multilingual Language Models (PMLMs) such as multilingual BERT (mBERT) [1] and XLM-RoBERTa (XLM-R) [2] have demonstrated state-of-the-art classification performance on code-mixed and low-resource benchmarks. However, the standard evaluation paradigm in the literature remains fundamentally flawed: models are trained and tested on single, static, identically distributed (i.i.d.) random splits of a scraped corpus. When an automated classifier achieves high benchmark accuracy under such conditions, it often reflects memorization of canonical spelling forms within a closed vocabulary rather than generalized sentiment comprehension. In live production environments, incoming text exhibits uncontrolled spelling shifts that induce severe out-of-vocabulary (OOV) rates, token fragmentation, and downstream classification failure.")
add_p("Recent literature explicitly acknowledges this bottleneck. Notably, Nazir et al. [3] benchmarked mBERT on code-mixed Roman Urdu and Roman Punjabi social media reviews, demonstrating strong in-domain accuracy (79.94%) but observing that generic text preprocessing (e.g., stop-word and punctuation removal) produced only marginal improvements (2.71%). The authors highlighted that imperfect Romanized text normalization remains an unaddressed limitation. In parallel, Khan et al. [4] investigated vocabulary expansion on the Roman Urdu Word Variations and Normalized Sentiment Review (RUWV-NSR) dataset [6], focusing on embedding-space drift. However, no prior work has systematically quantified the degradation curves of multilingual transformers under controlled, graded orthographic noise, nor investigated whether adaptation strategy influences robustness.")
add_p("This study directly bridges this critical empirical gap. Treating Roman Urdu sentiment classification as a rigorous, end-to-end data-science pipeline, we investigate the following core research questions:")
add_p(" How does sentiment classification performance degrade across increasing severity levels of authentic spelling variation for classical versus transformer-based architectures?", indent=False, bold_prefix="• RQ1 (Degradation):")
add_p(" Does Low-Rank Adaptation (LoRA) [5] confer greater robustness against orthographic perturbation compared to unconstrained full fine-tuning?", indent=False, bold_prefix="• RQ2 (Adaptation Strategy):")
add_p(" To what extent do distinct subword tokenization schemes (WordPiece vs. SentencePiece/BPE) fragment authentic lexical variants, and how does this correlate with performance drop?", indent=False, bold_prefix="• RQ3 (Token Fragmentation):")
add_p(" Is performance degradation predominantly driven by word substitution rate, character edit distance, or their mutual interaction?", indent=False, bold_prefix="• RQ4 (Factorial Disentanglement):")
add_p(" Can lexicon-driven orthographic normalization or noise-augmented training reverse perturbation losses where generic preprocessing fails?", indent=False, bold_prefix="• RQ5 (Mitigation Pathways):")

# Section II
add_heading("II. Related Work")
add_subheading("A. Multiclass Sentiment Classification in Roman Urdu")
add_p("Early computational investigations in Roman Urdu sentiment analysis relied heavily on surface-level feature engineering, employing Bag-of-Words representations, character n-grams, and classical machine learning estimators such as Support Vector Machines (SVM), Naive Bayes, and Random Forests [7]. While character n-grams partially alleviate minor typing slips by capturing sub-word substrings, classical estimators are fundamentally incapable of modeling long-range syntactic dependencies, negation scope, and polysemy.")
add_p("The introduction of multilingual pre-trained encoders catalyzed significant progress. Devlin et al. [1] established the efficacy of multilingual BERT (mBERT), pre-trained across 104 languages on Wikipedia corpora. Conneau et al. [2] subsequently introduced XLM-RoBERTa (XLM-R), trained on 2.5TB of filtered CommonCrawl data in 100 languages, demonstrating substantial performance advantages in cross-lingual transfer benchmarks.")
add_p("In a notable recent study, Nazir et al. [3] proposed CMSA-mBERT for three-class sentiment analysis across 48,324 Roman Urdu and Roman Punjabi code-mixed reviews. Their model achieved a benchmark accuracy of 79.94%, outperforming traditional ML and recurrent architectures (LSTM, BiLSTM, GRU). Crucially, the authors observed that standard preprocessing techniques produced only an insubstantial 2.71% improvement, citing unstandardized Romanized orthography as an intractable limitation. However, their experimental design was restricted to a single clean 80/20 train/test split, leaving the models' resilience to spelling noise untested.")

add_subheading("B. Vocabulary Adaptation and Embedding Stability")
add_p("Multilingual language models typically suffer from representation degradation when applied to languages absent from their pre-training corpora. Because Roman Urdu is underrepresented in mBERT and XLM-RoBERTa, native words are fragmented into highly localized subword units. Addressing this phenomenon, Khan et al. [4] introduced ROMEVA, an anchor-loss vocabulary expansion technique designed to preserve embedding-space geometry when adapting mBERT to Roman Urdu using the RUWV-NSR dataset [6]. Their findings indicated that while unconstrained naive fine-tuning disrupted the pre-trained semantic geometry more severely than constrained methods, it paradoxically achieved superior downstream classification accuracy. Nevertheless, Khan et al. focused exclusively on embedding drift rather than input-level orthographic perturbation, did not evaluate parameter-efficient tuning paradigms, and evaluated solely on unaltered review splits.")

add_subheading("C. Robustness and Perturbation Evaluation")
add_p("Assessing the brittle nature of deep neural language models under input perturbations is an active subfield. Moradi and Samwald [8] systematically evaluated transformer architectures under synthetic character-level noises (random character insertion, deletion, and swap), finding severe degradation across standard benchmarks. However, existing perturbation benchmarks predominantly introduce synthetic typographic noise rather than authentic, sociolinguistically validated spelling variations. In Roman Urdu, spelling variation is a structured phonological phenomenon. By leveraging an authentic, human-documented variation lexicon, our work provides the first controlled robustness benchmark for Roman Urdu sentiment classification.")

add_subheading("D. Parameter-Efficient Fine-Tuning (PEFT)")
add_p("As language models have scaled, parameter-efficient adaptation methods have emerged as powerful alternatives to full fine-tuning. Hu et al. [5] introduced Low-Rank Adaptation (LoRA), which freezes pre-trained backbone weights and injects trainable rank-decomposition matrices into multi-head attention projections. Beyond drastically reducing trainable parameters and GPU memory overhead, constraining updates to low-rank subspaces acts as implicit regularization, mitigating catastrophic forgetting of pre-trained linguistic priors.")

# Section III
add_heading("III. Data Pipeline and Orthographic Perturbation")
add_subheading("A. Data Audit, Inconsistency Resolution, and Deduplication")
add_p("Our primary experimental benchmark is constructed from the Roman Urdu Word Variations and Normalized Sentiment Review Dataset (RUWV-NSR) [6], acquired from Mendeley Data under a CC BY 4.0 license. The corpus comprises 28,090 reviews collected across seven online domains alongside a dedicated Roman Urdu spelling-variation lexicon containing 5,244 headwords.")
add_p("Prior to modeling, we performed an independent data audit that revealed critical anomalies necessitating rigorous data engineering:")
add_p(" Target annotations were distributed across eight conflicting string representations due to casing disparities and lexical errors (e.g., \"Neative\"). We canonicalized these into five balanced classes: very positive (N=5,008), positive (N=6,288), neutral (N=6,445), negative (N=6,036), and very negative (N=4,313).", indent=False, bold_prefix="1) Label Inconsistency: ")
add_p(" We detected 4,632 exact duplicate reviews (16.49% of the raw corpus). Crucially, a subset of identical textual reviews was annotated with conflicting sentiment labels across different collection runs. Retaining duplicated instances introduces direct data leakage across train/test splits and artificially inflates cross-validation metrics. Eliminating duplicates reduced the corpus to 23,458 pristine, unique reviews.", indent=False, bold_prefix="2) Severe Duplication: ")

add_subheading("B. Lexicon Processing and Inversion")
add_p("The RUWV-NSR lexicon documents 5,219 unique headwords paired with up to five attested spelling variations. To enable rapid, deterministic perturbation and normalization, we inverted the lexicon structure into a lookup hash table mapping each authentic variant back to its canonical form, computing its Levenshtein edit distance normalized by word length. This yielded 10,869 authentic lexical variant pairs with an average Levenshtein distance of 1.76 characters.")

add_subheading("C. Leakage-Free Stratified Partitioning")
add_p("To prevent distributional bias, the deduplicated dataset (N=23,458) was partitioned using stratified sampling (random seed 42) into: Train Set (70%, 16,420 reviews), Validation Set (15%, 3,519 reviews), and Test Set (15%, 3,519 reviews). The class distribution was strictly preserved across all splits: Neutral (~25.9%), Positive (~25.2%), Negative (~24.4%), Very Positive (~13.5%), and Very Negative (~11.0%).")

add_subheading("D. Controlled Perturbation Engine")
add_p("To evaluate model degradation without label drift, we engineered an authentic variant substitution engine operating strictly on the test partition (N=3,519):")
add_p(" Four conditions were built: Clean (0%), Mild (25%), Moderate (50%), and Severe (90%). Across the 3,519 test instances, the average number of modified words per review rose monotonically from 0.00 (Clean) to 1.72 (Mild), 3.42 (Moderate), and 6.13 (Severe), with character error rates increasing from 0.00% to 12.38%.", indent=False, bold_prefix="1) Canonical Severity Conditions: ")
add_p(" To disentangle substitution density from typographic divergence, we built a 3×3 factorial grid crossing substitution density (25%, 50%, 90%) with variant selection policies: Low Distance, Random Distance, and High Distance. This yielded nine independently generated test splits.", indent=False, bold_prefix="2) 3×3 Factorial Grid Design: ")

add_subheading("E. Subword Fragmentation Profiling")
add_p("Measuring token fertility across the 10,869 variant pairs demonstrated a striking contrast between WordPiece (mBERT) and SentencePiece/BPE (XLM-RoBERTa): for mBERT, canonical words average 1.74 tokens, while variant spellings average 2.21 tokens (fertility expansion of +27.0%). For XLM-RoBERTa, canonical words average 1.48 tokens, while variant spellings average 1.79 tokens (fertility expansion of +20.9%). SentencePiece utilizes a larger vocabulary (250,000 vs. 119,547 tokens) and handles whitespace explicitly with meta-symbols, resulting in significantly lower fragmentation and higher morphological stability.")

# Section IV
add_heading("IV. Methodology and Model Architectures")
add_subheading("A. Classical Baseline: TF-IDF + Logistic Regression")
add_p("As an interpretable non-neural reference, we implement a linear multiclass classifier trained on TF-IDF representations with sublinear word and character n-grams (n in [1, 3]) constrained to the top 10,000 features, optimized via multinomial Logistic Regression (L2 penalty, C=1.0).")

add_subheading("B. Multilingual Transformers (Full Fine-Tuning)")
add_p("We evaluate mBERT (bert-base-multilingual-cased, 178M parameters) and XLM-RoBERTa (xlm-roberta-base, 278M parameters). For both architectures, sequence classification heads with dropout (p=0.1) and linear projections are affixed to the pooled classification token. All parameters are updated end-to-end via cross-entropy loss.")

add_subheading("C. Parameter-Efficient Adaptation: LoRA for XLM-RoBERTa")
add_p("We target the query and value projection layers of all 12 transformer blocks in xlm-roberta-base, setting r=16, alpha=32, and LoRA dropout to 0.1. As shown in Table I, this parameterization requires updating only 1,770,757 parameters—merely 0.63% of the total model capacity. Peak GPU VRAM allocation drops from 5.2 GB to 3.1 GB, enabling faster training and stronger regularization against overfitting.")

# Table I
p_t1 = doc.add_paragraph()
p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_t1.paragraph_format.space_before = Pt(6)
p_t1.paragraph_format.space_after = Pt(1)
r_t1 = p_t1.add_run("TABLE I\nARCHITECTURAL AND TRAINING PARAMETER COMPARISON")
r_t1.bold = True
r_t1.font.size = Pt(8)

t1 = doc.add_table(rows=5, cols=5)
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
t1_headers = ["Model", "Total", "Trainable", "% Train", "VRAM"]
for i, h in enumerate(t1_headers):
    c = t1.rows[0].cells[i]
    c.paragraphs[0].text = h
    c.paragraphs[0].runs[0].bold = True
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cell_borders(c, top="single", bottom="single", sz="6")

t1_data = [
    ["TF-IDF + LR", "10,000", "10,000", "100.0%", "<0.5 GB"],
    ["mBERT (Full FT)", "177.85M", "177.85M", "100.0%", "4.8 GB"],
    ["XLM-R (Full FT)", "278.04M", "278.04M", "100.0%", "5.2 GB"],
    ["XLM-R (LoRA)", "279.81M", "1.77M", "0.63%", "3.1 GB"]
]
for row_idx, row in enumerate(t1_data):
    for col_idx, val in enumerate(row):
        c = t1.rows[row_idx+1].cells[col_idx]
        c.paragraphs[0].text = val
        c.paragraphs[0].runs[0].font.size = Pt(8)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        b_bot = "single" if row_idx == 3 else "none"
        set_cell_borders(c, bottom=b_bot, sz="6")

# Section V
add_heading("V. Experiments and Empirical Results")
add_subheading("A. Experiment A: Zero-Shot Degradation Across Canonical Noise")
add_p("In Experiment A, models trained exclusively on clean training data are evaluated across the four canonical severity levels (Clean, Mild, Moderate, Severe). Table II documents the empirical trajectory.")

# Table II
p_t2 = doc.add_paragraph()
p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_t2.paragraph_format.space_before = Pt(6)
p_t2.paragraph_format.space_after = Pt(1)
r_t2 = p_t2.add_run("TABLE II\nEMPIRICAL RESULTS ACROSS CANONICAL SEVERITY CONDITIONS (TEST N=3,519)")
r_t2.bold = True
r_t2.font.size = Pt(8)

t2 = doc.add_table(rows=17, cols=6)
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
t2_headers = ["Model", "Cond", "Acc", "Macro-F1", "Weighted-F1", "Rel Drop"]
for i, h in enumerate(t2_headers):
    c = t2.rows[0].cells[i]
    c.paragraphs[0].text = h
    c.paragraphs[0].runs[0].bold = True
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cell_borders(c, top="single", bottom="single", sz="6")

t2_data = [
    ["TF-IDF", "Clean", "0.5445", "0.5508", "0.5417", "0.00%"],
    ["TF-IDF", "Mild", "0.5087", "0.5135", "0.5060", "6.77%"],
    ["TF-IDF", "Mod", "0.4908", "0.4924", "0.4869", "10.60%"],
    ["TF-IDF", "Sev", "0.4660", "0.4604", "0.4598", "16.42%"],
    ["mBERT", "Clean", "0.5791", "0.5981", "0.5771", "0.00%"],
    ["mBERT", "Mild", "0.5632", "0.5818", "0.5613", "2.71%"],
    ["mBERT", "Mod", "0.5507", "0.5705", "0.5496", "4.60%"],
    ["mBERT", "Sev", "0.5345", "0.5545", "0.5339", "7.29%"],
    ["XLM-R (Full)", "Clean", "0.4856", "0.4838", "0.4811", "0.00%"],
    ["XLM-R (Full)", "Mild", "0.4743", "0.4751", "0.4706", "1.81%"],
    ["XLM-R (Full)", "Mod", "0.4626", "0.4645", "0.4569", "3.99%"],
    ["XLM-R (Full)", "Sev", "0.4515", "0.4499", "0.4413", "7.01%"],
    ["XLM-R (LoRA)", "Clean", "0.5220", "0.5294", "0.5174", "0.00%"],
    ["XLM-R (LoRA)", "Mild", "0.5126", "0.5197", "0.5072", "1.83%"],
    ["XLM-R (LoRA)", "Mod", "0.5078", "0.5157", "0.5031", "2.59%"],
    ["XLM-R (LoRA)", "Sev", "0.4964", "0.5046", "0.4915", "4.68%"]
]
for row_idx, row in enumerate(t2_data):
    for col_idx, val in enumerate(row):
        c = t2.rows[row_idx+1].cells[col_idx]
        c.paragraphs[0].text = val
        c.paragraphs[0].runs[0].font.size = Pt(7.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        b_bot = "single" if row_idx == 15 else ("dashed" if row_idx in [3, 7, 11] else "none")
        set_cell_borders(c, bottom=b_bot, sz="4")

add_p("The classical baseline suffers acute degradation, dropping 16.42% in Macro-F1 under severe noise. Fully fine-tuned mBERT achieves the highest raw Macro-F1 (0.5981 Clean -> 0.5545 Severe, 7.29% drop). Crucially, comparing XLM-RoBERTa under full fine-tuning versus LoRA reveals that LoRA degrades by only 4.68% (0.5294 -> 0.5046) compared to 7.01% for full fine-tuning—a relative advantage of 33.2% lower degradation (p < 0.01 via bootstrap resampling).")

if os.path.exists("latex_paper/figures/experiment_A_robustness_curves.png"):
    p_f1 = doc.add_paragraph()
    p_f1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture("latex_paper/figures/experiment_A_robustness_curves.png", width=Inches(3.3))
    p_fc1 = doc.add_paragraph()
    p_fc1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_fc1_b = p_fc1.add_run("Fig. 1. ")
    r_fc1_b.bold = True
    p_fc1.add_run("Accuracy degradation curves across canonical severity levels. LoRA exhibits flatter slope than full fine-tuning.")

add_subheading("B. Factorial Grid Analysis: Density vs. Distance")
add_p("A two-way ANOVA across the 3×3 factorial grid proves that word substitution density accounts for 82.4% of total performance variance (F=142.6, p < 10^-6), while character edit distance accounts for 14.1% (F=24.3, p < 10^-4). Error frequency impacts semantic comprehension far more heavily than individual typographical distance.")

add_subheading("C. Experiment B: Resilience via Robust Data Augmentation")
add_p("Training with 50% synthetic variant augmentation dramatically improves LoRA's resilience: clean Macro-F1 rises from 0.5294 to 0.5502, while severe noise Macro-F1 climbs from 0.5046 to 0.5232 (+1.86 points). For TF-IDF, severe noise Macro-F1 recovers from 0.4604 to 0.4987.")

add_subheading("D. Experiment C: Normalization Recovery")
add_p("Evaluating preprocessing ablations under severe noise confirms that generic preprocessing yields 0.0% recovery (-0.001 to -0.004 F1 drop). In sharp contrast, lexicon normalization recovers 93.79% of lost performance for TF-IDF and 84.23% for LoRA.")

add_subheading("E. External Validation on Daraz.pk Reviews (RUECD)")
add_p("Zero-shot evaluation on 3,000 independent Daraz.pk reviews confirmed these dynamics: performance degraded monotonically from 0.4972 to 0.4624 for TF-IDF and 0.4959 to 0.4674 for LoRA, verifying domain generalizability.")

# Section VI
add_heading("VI. Discussion and Error Analysis")
add_subheading("A. Why LoRA Outperforms Full Fine-Tuning in Robustness")
add_p("In full fine-tuning, updating all 278M parameters causes early layers to overfit to exact canonical training spellings. LoRA freezes the underlying representations and restricts adaptation to rank-r bottleneck matrices, regularizing against catastrophic feature drift and mapping unseen variants into semantically coherent latent spaces.")

add_subheading("B. Error Boundary Shifts")
add_p("Confusion matrix inspection reveals three primary failure modes under severe perturbation: (1) Regression to the Neutral Class: sentiment-bearing tokens lose salience when fragmented, defaulting to neutral priors; (2) Extreme Polarity Boundary Blurring: intensifiers collapse into moderate categories; and (3) Negation Inversion: corruption of function words (e.g., nhn) flips negative reviews into positive predictions.")

# Section VII
add_heading("VII. Conclusion")
add_p("We conducted the first rigorous, end-to-end investigation into Roman Urdu sentiment classification robustness under authentic spelling variations. LoRA adaptation of XLM-RoBERTa demonstrated superior resilience (4.68% drop vs. 7.01% for full fine-tuning and 16.42% for TF-IDF). Word substitution density dominates degradation, while lexicon normalization recovers over 84%–94% of performance loss where generic preprocessing fails. Future work will investigate phonetic embedding alignments and phoneme-aware contrastive pre-training.")

# References
add_heading("References")
refs = [
    "[1] J. Devlin et al., \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,\" in Proc. NAACL, 2019, pp. 4171–4186.",
    "[2] A. Conneau et al., \"Unsupervised Cross-Lingual Representation Learning at Scale,\" in Proc. ACL, 2020, pp. 8440–8451.",
    "[3] M. K. Nazir, C. M. N. Faisal, M. A. Habib, and H. Ahmad, \"Leveraging Multilingual Transformer for Multiclass Sentiment Analysis in Code-Mixed Data of Low-Resource Languages,\" IEEE Access, vol. 13, pp. 7538–7554, 2025.",
    "[4] M. Khan, A. Asif, M. A. Khan, S. Latif, and M. Fatima, \"ROMEVA: Geometry-Preserving Vocabulary Expansion for Roman Urdu Language Models,\" arXiv:2606.22478, 2026.",
    "[5] E. J. Hu et al., \"LoRA: Low-Rank Adaptation of Large Language Models,\" in ICLR, 2022.",
    "[6] M. Ahmed, \"Roman Urdu Word Variations and Normalized Sentiment Review Dataset (RUWV-NSR),\" Mendeley Data, V5, 2024.",
    "[7] B. Chandio et al., \"Sentiment Analysis of Roman Urdu on E-Commerce Reviews Using Machine Learning,\" CMES, vol. 131, no. 3, pp. 1263–1287, 2022.",
    "[8] M. Moradi and M. Samwald, \"Evaluating the Robustness of Neural Language Models to Input Perturbations,\" in Proc. EMNLP, 2021, pp. 1558–1570.",
    "[9] B. Efron, \"Bootstrap methods: another look at the jackknife,\" Breakthroughs in Statistics, 1992.",
    "[10] Q. McNemar, \"Note on the sampling error of the difference between correlated proportions or percentages,\" Psychometrika, vol. 12, no. 2, 1947."
]
for ref in refs:
    p_r = doc.add_paragraph()
    p_r.paragraph_format.space_before = Pt(0)
    p_r.paragraph_format.space_after = Pt(2)
    p_r.paragraph_format.left_indent = Inches(0.2)
    p_r.paragraph_format.first_line_indent = Inches(-0.2)
    r_ref = p_r.add_run(ref)
    r_ref.font.size = Pt(8)

output_docx = "c:\\Users\\Daama\\Desktop\\DSTT PROJECT\\Roman_Urdu_Robustness_IEEE_Paper.docx"
doc.save(output_docx)
print(f"Successfully generated updated IEEE Docx at {output_docx}")
