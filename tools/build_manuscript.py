"""
tools/build_manuscript.py
=========================
Assembles the revised manuscript, the supplementary material and the response letter as
Word documents from the Markdown sources in manuscript/ and the analysis outputs in
revision_outputs/. Tables are generated from the result CSV/JSON files, so every number in
a table comes straight from the experiment.

Usage
-----
  python tools/build_manuscript.py
"""

import json
import os
import re
import sys

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potato_rev import config as C  # noqa: E402

ROOT = C.PROJECT_ROOT
MS = os.path.join(ROOT, "manuscript")
OUT = os.path.join(ROOT, "revision_outputs")
TAB = os.path.join(OUT, "analysis", "tables")
FIG = os.path.join(OUT, "analysis", "figures")

ARCH = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
        "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "efficientnet_b3": "EfficientNet-B3",
        "vgg16": "VGG-16", "vgg19": "VGG-19"}
ACT = {"relu": "ReLU", "relu6": "ReLU6", "lrelu": "Leaky ReLU", "prelu": "PReLU", "mish": "Mish",
       "swish": "Swish", "gelu": "GELU"}

TITLE = ("Systematic Evaluation of Deep Convolutional Neural Networks and Activation Functions for Potato "
         "Tuber Disease Classification: A Leakage-Controlled Cross-Validation Study")
AUTHORS = [
    ("Zubair Najim Dost", "1", "Department of Computer Engineering, Bandırma Onyedi Eylül University, Bandırma, Türkiye. "
                                "ORCID: 0009-0008-8240-1225. E-mail: zubairdost@ogr.bandirma.edu.tr"),
    ("Serhat Kılıçarslan", "2", "Department of Software Engineering, Bandırma Onyedi Eylül University, Bandırma, Türkiye. "
                                 "ORCID: 0000-0001-9483-4425. E-mail: skilicarslan@bandirma.edu.tr"),
]
ABSTRACT = (
    "Tuber diseases of potato (Solanum tuberosum L.) reduce yield, grading quality and seed health, and image-based "
    "diagnosis with convolutional neural networks (CNNs) could support tuber inspection. Practitioners often replace "
    "the activation function of pretrained CNNs, but the benefit of this practice has not been tested rigorously, and "
    "published accuracies on the public potato tuber image set are inflated by data leakage. We audited this dataset "
    "(451 web-collected, expert-verified, laboratory-style images of harvested tubers in seven classes: black scurf, "
    "blackleg, common scab, dry rot, pink rot, healthy tubers and miscellaneous disorders), removed 12 duplicated and "
    "8 label-conflicting photographs, and evaluated 49 configurations, namely eight ImageNet-pretrained CNNs "
    "(ResNet-18/50, DenseNet-121, MobileNetV2, EfficientNet-B0/B3, VGG-16/19) combined with ReLU, Leaky ReLU, PReLU, "
    "Swish, Mish and GELU, using duplicate-aware five-fold cross-validation with nested learning-rate selection "
    "(1,268 training runs). The best configuration, DenseNet-121 with its native ReLU, reached 81.2 ± 2.8% accuracy "
    "(error rate 18.8%), 81.4% macro-precision, 80.6% macro-recall, 80.4 ± 2.8% macro-F1 and a macro-AUC of 0.957, "
    "but the architectures did not differ significantly. Replacing the native activation of batch-normalised "
    "networks with one from the other family (piecewise-linear versus smooth) reduced macro-F1 by 6.8 percentage "
    "points (p < 0.001), whereas replacements within a family had no effect, and no replacement improved performance "
    "significantly. ImageNet pre-training contributed 32 points of macro-F1 and data augmentation about one point. "
    "Healthy tubers were recognised almost perfectly, while blackleg and dry rot were most often confused. The study "
    "provides a leakage-free benchmark and practical guidance for automated tuber inspection.")
KEYWORDS = ("Potato tuber diseases; Deep learning; Convolutional neural networks; Activation functions; "
            "Transfer learning; Data leakage; Cross-validation; Seed potato inspection")

REFERENCES = [
    "Devaux A, Goffart J-P, Petsakos A, Kromann P, Gatto M, Okello J, Suarez V, Hareau G (2020) Global food security, contributions from sustainable potato agri-food systems. In: Campos H, Ortiz O (eds) The Potato Crop. Springer, Cham, pp 3–35. https://doi.org/10.1007/978-3-030-28683-5_1",
    "Fiers M, Edel-Hermann V, Chatot C, Le Hingrat Y, Alabouvette C, Steinberg C (2012) Potato soil-borne diseases. A review. Agron Sustain Dev 32:93–132. https://doi.org/10.1007/s13593-011-0035-z",
    "Charkowski A, Sharma K, Parker ML, Secor GA, Elphinstone J (2020) Bacterial diseases of potato. In: Campos H, Ortiz O (eds) The Potato Crop. Springer, Cham, pp 351–388. https://doi.org/10.1007/978-3-030-28683-5_10",
    "Tsror L (2010) Biology, epidemiology and management of Rhizoctonia solani on potato. J Phytopathol 158:649–658. https://doi.org/10.1111/j.1439-0434.2010.01671.x",
    "Dees MW, Wanner LA (2012) In search of better management of potato common scab. Potato Res 55:249–268. https://doi.org/10.1007/s11540-012-9206-9",
    "Toth IK, van der Wolf JM, Saddler G, Lojkowska E, Hélias V, Pirhonen M, Tsror L, Elphinstone JG (2011) Dickeya species: an emerging problem for potato production in Europe. Plant Pathol 60:385–399. https://doi.org/10.1111/j.1365-3059.2011.02427.x",
    "Secor GA, Gudmestad NC (1999) Managing fungal diseases of potato. Can J Plant Pathol 21:213–221. https://doi.org/10.1080/07060669909501184",
    "ElMasry G, Cubero S, Moltó E, Blasco J (2012) In-line sorting of irregular potatoes by using automated computer-based machine vision system. J Food Eng 112:60–68. https://doi.org/10.1016/j.jfoodeng.2012.03.027",
    "Su Q, Kondo N, Li M, Sun H, Al Riza DF, Habaragamuwa H (2018) Potato quality grading based on machine vision with 3D shape analysis. Comput Electron Agric 152:261–268. https://doi.org/10.1016/j.compag.2018.07.012",
    "Mohanty SP, Hughes DP, Salathé M (2016) Using deep learning for image-based plant disease detection. Front Plant Sci 7:1419. https://doi.org/10.3389/fpls.2016.01419",
    "Ferentinos KP (2018) Deep learning models for plant disease detection and diagnosis. Comput Electron Agric 145:311–318. https://doi.org/10.1016/j.compag.2018.01.009",
    "Kamilaris A, Prenafeta-Boldú FX (2018) Deep learning in agriculture: a survey. Comput Electron Agric 147:70–90. https://doi.org/10.1016/j.compag.2018.02.016",
    "Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L (2009) ImageNet: a large-scale hierarchical image database. In: Proc IEEE Conf Comput Vis Pattern Recognit (CVPR), pp 248–255. https://doi.org/10.1109/CVPR.2009.5206848",
    "Barbedo JGA (2018) Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. Comput Electron Agric 153:46–53. https://doi.org/10.1016/j.compag.2018.08.013",
    "Kornblith S, Shlens J, Le QV (2019) Do better ImageNet models transfer better? In: Proc IEEE/CVF Conf Comput Vis Pattern Recognit (CVPR), pp 2661–2671",
    "Ramachandran P, Zoph B, Le QV (2017) Searching for activation functions. arXiv:1710.05941",
    "Misra D (2020) Mish: a self regularized non-monotonic activation function. In: Proc British Machine Vision Conference (BMVC). arXiv:1908.08681",
    "Hendrycks D, Gimpel K (2016) Gaussian error linear units (GELUs). arXiv:1606.08415",
    "Dubey SR, Singh SK, Chaudhuri BB (2022) Activation functions in deep learning: a comprehensive survey and benchmark. Neurocomputing 503:92–108. https://doi.org/10.1016/j.neucom.2022.06.111",
    "Luong HH (2024) Improving potato diseases classification based on custom ConvNeXtSmall and combine with the explanation model. Int J Adv Comput Sci Appl 15(4). https://thesai.org/Publications/ViewPaper?Volume=15&Issue=4&Code=IJACSA&SerialNo=121",
    "Prasetyo SY (2024) Overcoming overfitting in CNN models for potato disease classification using data augmentation. Jurnal EMACS 6(3):179–184. https://doi.org/10.21512/emacsjournal.v6i3.11840",
    "Kapoor S, Narayanan A (2023) Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4:100804. https://doi.org/10.1016/j.patter.2023.100804",
    "Barz B, Denzler J (2020) Do we train on test data? Purging CIFAR of near-duplicates. J Imaging 6:41. https://doi.org/10.3390/jimaging6060041",
    "Singh D, Jain N, Jain P, Kayal P, Kumawat S, Batra N (2020) PlantDoc: a dataset for visual plant disease detection. In: Proc 7th ACM IKDD CoDS and 25th COMAD, pp 249–253. https://doi.org/10.1145/3371158.3371196",
    "Dutta M, Gupta D, Gulzar Y, et al. (2024) Leveraging Inception V3 for precise early and late blight disease classification in potato crops. Traitement du Signal 41(2):705–715. https://doi.org/10.18280/ts.410213",
    "Gulzar Y (2024) Enhancing soybean classification with modified inception model: a transfer learning approach. Emir J Food Agric 36:1–9. https://doi.org/10.3897/ejfa.2024.122928",
    "Seelwal P, Dhiman P, Gulzar Y, Kaur A, Wadhwa S, Onn CW (2024) A systematic review of deep learning applications for rice disease diagnosis: current trends and future directions. Front Comput Sci 6:1452961. https://doi.org/10.3389/fcomp.2024.1452961",
    "Oppenheim D, Shani G (2017) Potato disease classification using convolution neural networks. Adv Anim Biosci 8(2):244–249. https://doi.org/10.1017/S2040470017001376",
    "Oppenheim D, Shani G, Erlich O, Tsror L (2019) Using deep learning for image-based potato tuber disease detection. Phytopathology 109:1083–1087. https://doi.org/10.1094/PHYTO-08-18-0288-R",
    "Marino S, Beauseroy P, Smolarz A (2019) Deep learning-based method for classifying and localizing potato blemishes. In: Proc 8th Int Conf Pattern Recognit Appl Methods (ICPRAM), pp 107–117",
    "Nair V, Hinton GE (2010) Rectified linear units improve restricted Boltzmann machines. In: Proc 27th Int Conf Mach Learn (ICML), pp 807–814",
    "Maas AL, Hannun AY, Ng AY (2013) Rectifier nonlinearities improve neural network acoustic models. In: ICML Workshop on Deep Learning for Audio, Speech and Language Processing",
    "He K, Zhang X, Ren S, Sun J (2015) Delving deep into rectifiers: surpassing human-level performance on ImageNet classification. In: Proc IEEE Int Conf Comput Vis (ICCV), pp 1026–1034. https://doi.org/10.1109/ICCV.2015.123",
    "Elfwing S, Uchibe E, Doya K (2018) Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Netw 107:3–11. https://doi.org/10.1016/j.neunet.2017.12.012",
    "Potato Diseases Datasets (n.d.) Kaggle, user mukaffimoin. https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets. Accessed 13 October 2026",
    "Islam S, Afrin T (2025) PotatoCare: deep learning based potato disease dataset. Mendeley Data, V2. https://doi.org/10.17632/7vm7xskfg4.2",
    "Simonyan K, Zisserman A (2015) Very deep convolutional networks for large-scale image recognition. In: Int Conf Learn Represent (ICLR). arXiv:1409.1556",
    "He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. In: Proc IEEE Conf Comput Vis Pattern Recognit (CVPR), pp 770–778. https://doi.org/10.1109/CVPR.2016.90",
    "Huang G, Liu Z, van der Maaten L, Weinberger KQ (2017) Densely connected convolutional networks. In: Proc IEEE Conf Comput Vis Pattern Recognit (CVPR), pp 4700–4708. https://doi.org/10.1109/CVPR.2017.243",
    "Sandler M, Howard A, Zhu M, Zhmoginov A, Chen L-C (2018) MobileNetV2: inverted residuals and linear bottlenecks. In: Proc IEEE/CVF Conf Comput Vis Pattern Recognit (CVPR), pp 4510–4520. https://doi.org/10.1109/CVPR.2018.00474",
    "Tan M, Le QV (2019) EfficientNet: rethinking model scaling for convolutional neural networks. In: Proc 36th Int Conf Mach Learn (ICML), pp 6105–6114",
    "Loshchilov I, Hutter F (2019) Decoupled weight decay regularization. In: Int Conf Learn Represent (ICLR). arXiv:1711.05101",
    "Goyal P, Dollár P, Girshick R, Noordhuis P, Wesolowski L, Kyrola A, Tulloch A, Jia Y, He K (2017) Accurate, large minibatch SGD: training ImageNet in 1 hour. arXiv:1706.02677",
    "Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z (2016) Rethinking the Inception architecture for computer vision. In: Proc IEEE Conf Comput Vis Pattern Recognit (CVPR), pp 2818–2826. https://doi.org/10.1109/CVPR.2016.308",
    "Demšar J (2006) Statistical comparisons of classifiers over multiple data sets. J Mach Learn Res 7:1–30",
    "Holm S (1979) A simple sequentially rejective multiple test procedure. Scand J Stat 6:65–70",
    "Paszke A, Gross S, Massa F, et al. (2019) PyTorch: an imperative style, high-performance deep learning library. In: Adv Neural Inf Process Syst 32 (NeurIPS), pp 8024–8035",
    "Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D (2017) Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proc IEEE Int Conf Comput Vis (ICCV), pp 618–626. https://doi.org/10.1109/ICCV.2017.74",
    "Ioffe S, Szegedy C (2015) Batch normalization: accelerating deep network training by reducing internal covariate shift. In: Proc 32nd Int Conf Mach Learn (ICML), pp 448–456",
    "Varma S, Simon R (2006) Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7:91. https://doi.org/10.1186/1471-2105-7-91",
    "Yosinski J, Clune J, Bengio Y, Lipson H (2014) How transferable are features in deep neural networks? In: Adv Neural Inf Process Syst 27 (NeurIPS), pp 3320–3328",
    "Gulzar Y, Ünal Z (2025) Optimizing pear leaf disease detection through PL-DenseNet. Appl Fruit Sci 67:40. https://doi.org/10.1007/s10341-025-01265-2",
    "Gulzar Y (2025) PapNet: an AI-driven approach for early detection and classification of papaya leaf diseases. Appl Fruit Sci 67:256. https://doi.org/10.1007/s10341-025-01466-9",
    "Gulzar Y (2025) Papaya leaf disease classification using pre-trained deep learning models: a comparative study. Appl Fruit Sci 67:287. https://doi.org/10.1007/s10341-025-01533-1",
    "Gulzar Y, Ünal Z (2025) Time-sensitive bruise detection in plums using PlmNet with transfer learning. Procedia Comput Sci 257:127–132. https://doi.org/10.1016/j.procs.2025.03.019",
    "Gulzar Y, Ünal Z, Kızıldeniz T, Umar UM (2024) Deep learning-based classification of alfalfa varieties: a comparative study using a custom leaf image dataset. MethodsX 13:103051. https://doi.org/10.1016/j.mex.2024.103051",
]

FIGURES = {
    "duplicates": ("fig_duplicates.png", "Examples of repeated photographs found by the data audit. Each column shows the two copies; red labels mark photographs that carry two different disease labels (excluded), black labels mark repeated photographs with the same label (one copy kept)."),
    "framework": ("fig_framework.png", "Overview of the evaluation protocol: data audit and curation, duplicate-aware fold assignment, nested learning-rate selection on the validation fold, and a single evaluation of the selected model on the test fold of each rotation."),
    "heatmap": ("heatmap_f1.png", "Test macro-F1 (%) of all 49 architecture–activation configurations (mean ± SD over five rotations). An asterisk marks each network's native activation."),
    "swap": ("swap_deltas.png", "Paired change in test macro-F1 when a network's native activation is replaced (mean ± standard error over five rotations). Blue: replacement from the same family as the native activation (piecewise-linear: ReLU, ReLU6, Leaky ReLU, PReLU; smooth: Swish, Mish, GELU); red: replacement from the other family."),
    "boxplots": ("boxplots_f1.png", "Distribution of test macro-F1 by architecture (left; each box pools six activations × five rotations) and by activation function (right; each box pools eight architectures × five rotations). Diamonds mark means."),
    "confusion": ("confusion_top3.png", "Confusion matrices of the three best configurations, pooled over the five test folds (each of the 431 images is predicted once by the model that did not see it in training). Cell shading is normalised by row (recall)."),
    "roc": ("roc_top3.png", "One-versus-rest ROC curves and per-class AUC of the three best configurations, from the pooled out-of-fold predictions."),
    "curves_es": ("learning_curves_early_stopped.png", "Training and validation loss and accuracy of the selected runs with the native activation on rotation 1, with early stopping. The dashed line marks the restored checkpoint (lowest validation loss)."),
    "curves_100": ("learning_curves_100_epochs.png", "Training and validation loss and accuracy over the full 100 epochs without early stopping (rotation 1, native activation, learning rate most often selected for the architecture). The dashed line marks the epoch of lowest validation loss."),
    "lr": ("lr_sensitivity.png", "Mean test macro-F1 over five rotations at each candidate learning rate (descriptive; model selection used validation loss only). Solid lines: native activation."),
    "gradcam": ("gradcam_densenet121_relu.png", "Grad-CAM heat maps of DenseNet-121 + ReLU for test images of rotation 1: for each class, the original image and the heat map of the predicted class (green: correct prediction; red: misclassification with the predicted class named)."),
}

TABLE_CAPTIONS = {
    "prior_work": "Studies on image-based potato tuber disease classification and on the dataset used here",
    "dataset": "Classes of the dataset, causal agents, diagnostic symptoms, and numbers of images before and after curation and per cross-validation fold",
    "architectures": "Backbone architectures: design principle, size, cost at 224 × 224 pixels, native activation and number of replaced activation sites",
    "augmentation": "Training images per class and on-the-fly augmented views presented during training",
    "system": "Hardware and software used for training and for the inference benchmark",
    "main": "Test macro-F1 (%) of every architecture–activation configuration, mean ± SD over five rotations (* native activation; best value per architecture in bold)",
    "stats": "Statistical tests on test macro-F1",
    "perclass": "Per-class precision, recall, specificity, F1-score and AUC of the three best configurations (pooled out-of-fold predictions, n = 431)",
    "ablation": "Ablation studies with the native activation: macro-F1 (%) with the full protocol, without data augmentation and without ImageNet pre-training (mean over five rotations, nested learning-rate selection in every case)",
    "compute": "Computational cost of each backbone with its native activation (inference on an RTX 3050 Ti laptop GPU and an Intel Core i5-11320H CPU; training on an A100 GPU)",
}


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)")


def add_runs(par, text: str, size: float = None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = par.add_run(part[1:-1])
            run.italic = True
        else:
            run = par.add_run(part)
        if size:
            run.font.size = Pt(size)
    return par


def setup(doc: Document):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.paragraph_format.line_spacing = 2.0
    st.paragraph_format.space_after = Pt(0)
    for name in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        s = doc.styles[name]
        s.font.name = "Times New Roman"
        s.font.color.rgb = None
    doc.styles["Heading 1"].font.size = Pt(14)
    doc.styles["Heading 2"].font.size = Pt(12)
    for sec in doc.sections:
        sec.left_margin = sec.right_margin = Cm(2.5)
        sec.top_margin = sec.bottom_margin = Cm(2.5)


def add_table(doc: Document, df: pd.DataFrame, caption: str, number: int, note: str = None, font: float = 8.5,
              bold_cells=None):
    cap = doc.add_paragraph()
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.space_before = Pt(8)
    r = cap.add_run(f"Table {number} ")
    r.bold = True
    cap.add_run(caption)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(col))
        run.bold = True
        run.font.size = Pt(font)
    for i, row in enumerate(df.itertuples(index=False)):
        cells = table.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run("" if pd.isna(val) else remap(str(val)))
            run.font.size = Pt(font)
            if bold_cells and (i, j) in bold_cells:
                run.bold = True
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.0
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        add_runs(p, note, size=9)
    doc.add_paragraph()


def add_figure(doc: Document, key: str, number: int, width_cm: float = 16.0):
    fname, caption = FIGURES[key]
    path = os.path.join(FIG, fname)
    if not os.path.exists(path):
        doc.add_paragraph(f"[Figure {number} missing: {fname}]")
        return
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"Fig. {number} ")
    r.bold = True
    p.add_run(caption)
    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Tables from the results
# ---------------------------------------------------------------------------

def pm(mean, sd, scale=100.0, nd=1):
    return f"{scale * mean:.{nd}f} ± {scale * sd:.{nd}f}"


def table_prior_work() -> pd.DataFrame:
    rows = [
        ["Oppenheim and Shani (2017) [28]; Oppenheim et al. (2019) [29]", "Own expert-labelled tuber images (low-cost RGB); 4 diseases + healthy",
         "Train/test split", "CNN (VGG-type) trained on tuber images", "High classification accuracy", "Data not public; no comparison of activations"],
        ["Marino et al. (2019) [30]", "Own tuber images; blemish classes", "Train/test split", "CNN with localisation",
         "Average F1 ≈ 84% (as cited in [20])", "Grading-line blemishes, not the diseases studied here"],
        ["Luong (2024) [20]", "This dataset (451 images, 7 classes)", "Augmentation to 5,833 images before a random 80/10/10 split",
         "Custom ConvNeXt-Small, other CNNs", "98.97% test accuracy", "Augmented copies of a photograph in training and test data; duplicates not controlled; single split"],
        ["Prasetyo (2024) [21]", "This dataset (451 images, 7 classes)", "80/10/10 split before augmentation",
         "VGG-16/19, Xception, Inception V3", "≈73–75% test accuracy (VGG-16)", "Single 45-image test split; duplicates not controlled; no variance or tests"],
        ["This study", "This dataset after audit (431 images, 7 classes)", "Duplicate-aware 5-fold CV, nested LR selection",
         "8 backbones × 6 activations (+ ReLU6), 1,268 runs", "81.2 ± 2.8% accuracy, 80.4 ± 2.8% macro-F1",
         "Single, web-sourced laboratory-style dataset"],
    ]
    return pd.DataFrame(rows, columns=["Study", "Data", "Evaluation protocol", "Models", "Reported result", "Main limitation"])


def table_dataset() -> pd.DataFrame:
    with open(C.MANIFEST_PATH) as f:
        m = json.load(f)
    info = {
        "Black Scurf": ("Rhizoctonia solani (fungus)", "Black sclerotia on the tuber skin"),
        "Blackleg": ("Pectobacterium and Dickeya spp. (bacteria)", "Dark, wet decay spreading from the stolon end; internal discoloration"),
        "Common Scab": ("Streptomyces scabies and related spp. (actinobacteria)", "Corky, raised or pitted lesions on the skin"),
        "Dry Rot": ("Fusarium spp. (fungi)", "Sunken, wrinkled, dry decay with internal cavities"),
        "Healthy Potatoes": ("–", "Intact skin without lesions"),
        "Miscellaneous": ("Various (not documented by the source)", "Other rots, physiological disorders, mechanical damage"),
        "Pink Rot": ("Phytophthora erythroseptica (oomycete)", "Rubbery tissue that turns pink when the cut surface is exposed to air"),
    }
    rows = []
    for c, name in enumerate(m["class_names"]):
        imgs = [x for x in m["images"] if x["label"] == c]
        kept = [x for x in imgs if x["status"] == "kept"]
        folds = [sum(1 for x in kept if x["fold"] == k) for k in range(m["n_folds"])]
        rows.append([name, *info[name], len(imgs), len(kept), " / ".join(map(str, folds))])
    total_raw = len(m["images"])
    total_kept = sum(1 for x in m["images"] if x["status"] == "kept")
    folds = [sum(1 for x in m["images"] if x["status"] == "kept" and x["fold"] == k) for k in range(m["n_folds"])]
    rows.append(["Total", "", "", total_raw, total_kept, " / ".join(map(str, folds))])
    return pd.DataFrame(rows, columns=["Class", "Causal agent", "Diagnostic symptoms", "Raw images", "Curated images", "Per fold (1/2/3/4/5)"])


def benchmark() -> pd.DataFrame:
    with open(os.path.join(OUT, "benchmark_NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU.json")) as f:
        return pd.DataFrame(json.load(f)["records"])


def table_architectures() -> pd.DataFrame:
    b = benchmark()
    design = {
        "resnet18": "Residual (basic blocks)", "resnet50": "Residual (bottleneck blocks)",
        "densenet121": "Dense connectivity, feature reuse", "mobilenet_v2": "Inverted residuals, depthwise convolutions",
        "efficientnet_b0": "Compound scaling, MBConv + SE", "efficientnet_b3": "Compound scaling, MBConv + SE (larger)",
        "vgg16": "Plain 3 × 3 stack, no normalisation", "vgg19": "Plain 3 × 3 stack, no normalisation (deeper)",
    }
    sites = {"resnet18": 9, "resnet50": 17, "densenet121": 121, "mobilenet_v2": 35,
             "efficientnet_b0": 49, "efficientnet_b3": 78, "vgg16": 15, "vgg19": 18}
    rows = []
    for a in C.ARCHITECTURES:
        r = b[(b.arch == a) & b.native].iloc[0]
        rows.append([ARCH[a], design[a], f"{r.params_m:.1f}", f"{r.gmacs:.2f}", ACT[C.NATIVE_ACTIVATION[a]],
                     "Yes" if not a.startswith("vgg") else "No", sites[a]])
    return pd.DataFrame(rows, columns=["Architecture", "Design principle", "Parameters (M)", "GMAC / image",
                                       "Native activation", "Batch norm.", "Activation sites"])


def table_augmentation() -> pd.DataFrame:
    with open(C.MANIFEST_PATH) as f:
        m = json.load(f)
    sel = pd.read_csv(os.path.join(TAB, "per_fold_selected.csv"))
    epochs = sel["epochs_run"].mean()
    kept = [x for x in m["images"] if x["status"] == "kept"]
    rows = []
    for c, name in enumerate(m["class_names"]):
        per_rot = [sum(1 for x in kept if x["label"] == c and x["fold"] not in (k, (k + 1) % 5)) for k in range(5)]
        n = sum(per_rot) / 5
        rows.append([name, f"{min(per_rot)}–{max(per_rot)}", f"{n:.1f}", f"{n * epochs:,.0f}", f"{n * 100:,.0f}"])
    return pd.DataFrame(rows, columns=["Class", "Original training images per rotation", "Mean",
                                       f"Augmented views per run (mean {epochs:.1f} epochs)", "Augmented views per run (100 epochs)"])


def table_system() -> pd.DataFrame:
    with open(os.path.join(OUT, "system_specs.json")) as f:
        t = json.load(f)
    with open(os.path.join(OUT, "benchmark_NVIDIA_GeForce_RTX_3050_Ti_Laptop_GPU.json")) as f:
        b = json.load(f)["system"]
    rows = [
        ["Purpose", "Training and evaluation (all 1,268 runs)", "Inference benchmark (Table 10)"],
        ["Platform", "Google Colab Pro, " + t["os"], "Laptop, Windows 11"],
        ["GPU", f"{t['gpu']} ({t['gpu_vram_gb']} GB)", f"{b['gpu']} ({b['gpu_vram_gb']} GB)"],
        ["CPU", f"{t['cpu']}, {t['cpu_logical_cores']} vCPUs", f"Intel Core i5-11320H @ 3.20 GHz, {b['cpu_logical_cores']} threads ({b['cpu_threads']} used)"],
        ["RAM", f"{t['ram_gb']} GB", f"{b['ram_gb']} GB"],
        ["Python / PyTorch / torchvision", f"{t['python']} / {t['pytorch']} / {t['torchvision']}", f"{b['python']} / {b['pytorch']} / {b['torchvision']}"],
        ["CUDA / cuDNN", f"{t['cuda']} / {t['cudnn']}", f"{b['cuda']} / {b['cudnn']}"],
        ["Numerical precision", "32-bit floating point (TF32 matrix units enabled)", "32-bit floating point"],
    ]
    return pd.DataFrame(rows, columns=["Item", "Training system", "Benchmark system"])


def table_main():
    s = pd.read_csv(os.path.join(TAB, "all_configurations.csv"))
    acts = C.ACTIVATIONS + ["relu6"]
    rows, bold = [], set()
    for i, a in enumerate(C.ARCHITECTURES):
        g = s[s.arch == a]
        best = g.f1_macro_mean.max()
        row = [ARCH[a]]
        for j, act in enumerate(acts):
            r = g[g.act == act]
            if r.empty:
                row.append("–")
                continue
            r = r.iloc[0]
            row.append(pm(r.f1_macro_mean, r.f1_macro_std) + ("*" if r.native else ""))
            if r.f1_macro_mean == best:
                bold.add((i, j + 1))
        rows.append(row)
    means = s[s.act.isin(C.ACTIVATIONS)].groupby("act").f1_macro_mean.mean()
    rows.append(["Mean over architectures"] + [f"{100 * means[a]:.1f}" for a in C.ACTIVATIONS] + ["–"])
    return pd.DataFrame(rows, columns=["Architecture"] + [ACT[a] for a in acts]), bold


def table_stats() -> pd.DataFrame:
    fam = pd.read_csv(os.path.join(TAB, "stats_activation_family.csv"))
    txt = open(os.path.join(OUT, "analysis", "summary.txt"), encoding="utf-8").read()
    fr_act = re.search(r"All architectures: Friedman chi2=([\d.]+), df=(\d+), p=([\d.e-]+), Kendall W=([\d.]+)", txt).groups()
    fr_arch = re.search(r"All activations: Friedman chi2=([\d.]+), df=(\d+), p=([\d.e-]+), Kendall W=([\d.]+)", txt).groups()
    bn = fam[fam.networks == "batch-normalised"].iloc[0]
    vgg = fam[fam.networks == "BN-free (VGG)"].iloc[0]
    fmt = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    rows = [
        ["Effect of activation (6 activations; 40 blocks: architecture × rotation)", "Friedman",
         f"χ² = {float(fr_act[0]):.1f}, df = {fr_act[1]}, W = {float(fr_act[3]):.2f}", fmt(float(fr_act[2])),
         "Significant pairs after Holm: ReLU > PReLU, Leaky ReLU > PReLU"],
        ["Effect of architecture (8 architectures; 30 blocks: activation × rotation)", "Friedman",
         f"χ² = {float(fr_arch[0]):.1f}, df = {fr_arch[1]}, W = {float(fr_arch[3]):.2f}", fmt(float(fr_arch[2])),
         "No significant pair after Holm"],
        ["BN networks: swap to the other activation family vs native (30 blocks)", "Wilcoxon",
         f"Δ = {bn.cross_family_delta_pp:+.1f} ± {bn.cross_sd_pp:.1f} pp", fmt(bn.p_cross_vs_0), "Consistent loss"],
        ["BN networks: swap within the same family vs native (30 blocks)", "Wilcoxon",
         f"Δ = {bn.within_family_delta_pp:+.1f} ± {bn.within_sd_pp:.1f} pp", fmt(bn.p_within_vs_0), "No effect"],
        ["BN networks: cross-family vs within-family swap", "Wilcoxon", "", fmt(bn.p_cross_vs_within), ""],
        ["VGG: swap to the other family vs native (10 blocks)", "Wilcoxon",
         f"Δ = {vgg.cross_family_delta_pp:+.1f} ± {vgg.cross_sd_pp:.1f} pp", fmt(vgg.p_cross_vs_0), "No effect"],
        ["VGG: swap within the same family vs native (10 blocks)", "Wilcoxon",
         f"Δ = {vgg.within_family_delta_pp:+.1f} ± {vgg.within_sd_pp:.1f} pp", fmt(vgg.p_within_vs_0), "Driven by PReLU"],
        ["Individual swaps vs native (41 comparisons, 5 rotations each)", "Wilcoxon + Holm", "10/41 higher mean F1 (max +2.6 pp)",
         "all ≥ 0.06", "None significant; minimum attainable p = 0.0625"],
    ]
    return pd.DataFrame(rows, columns=["Comparison", "Test", "Effect", "p", "Remark"])


def table_perclass() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(TAB, "per_class_top3.csv"))
    rows = []
    for _, r in d.iterrows():
        rows.append([r["model"], r["class"], int(r["support"]), f"{100 * r.precision:.1f}", f"{100 * r.recall:.1f}",
                     f"{100 * r.specificity:.1f}", f"{100 * r.f1:.1f}", f"{r.auc:.3f}"])
    return pd.DataFrame(rows, columns=["Model", "Class", "n", "Precision", "Recall", "Specificity", "F1", "AUC"])


def table_ablation() -> pd.DataFrame:
    a = pd.read_csv(os.path.join(TAB, "ablations.csv"))
    rows = []
    for arch in C.ARCHITECTURES:
        lab = ARCH[arch]
        na = a[(a.arch == lab) & (a.ablation == "no augmentation")]
        sc = a[(a.arch == lab) & (a.ablation == "no ImageNet pre-training")]
        if na.empty or sc.empty:
            continue
        na, sc = na.iloc[0], sc.iloc[0]
        rows.append([lab, ACT[C.NATIVE_ACTIVATION[arch]], f"{na.full_f1:.1f}", f"{na.ablated_f1:.1f} ({na.delta_pp:+.1f})",
                     f"{sc.ablated_f1:.1f} ({sc.delta_pp:+.1f})"])
    na = a[a.ablation == "no augmentation"]
    sc = a[a.ablation == "no ImageNet pre-training"]
    rows.append(["Mean", "", f"{na.full_f1.mean():.1f}", f"{na.ablated_f1.mean():.1f} ({na.delta_pp.mean():+.1f})",
                 f"{sc.ablated_f1.mean():.1f} ({sc.delta_pp.mean():+.1f})"])
    return pd.DataFrame(rows, columns=["Architecture", "Activation", "Full protocol", "Without augmentation (Δ)",
                                       "Without pre-training (Δ)"])


def table_compute() -> pd.DataFrame:
    b = benchmark()
    cost = pd.read_csv(os.path.join(TAB, "training_cost.csv")).set_index("arch")
    rows = []
    for a in C.ARCHITECTURES:
        r = b[(b.arch == a) & b.native].iloc[0]
        c = cost.loc[ARCH[a]]
        rows.append([ARCH[a], f"{r.params_m:.1f}", f"{r.gmacs:.2f}", f"{r.size_mb:.1f}", f"{r.gpu_ms:.1f}",
                     f"{r.cpu_ms:.0f}", f"{r.gpu_img_per_s_b32:.0f}", f"{c.epoch_time_s:.2f}", f"{c.peak_mem_mb / 1024:.1f}"])
    return pd.DataFrame(rows, columns=["Architecture", "Params (M)", "GMAC", "Size (MB)", "GPU latency (ms)",
                                       "CPU latency (ms)", "GPU throughput (img/s, batch 32)", "Training s/epoch (A100)",
                                       "Peak training memory (GB)"])


TABLES = {
    "prior_work": (table_prior_work, "CV cross-validation; LR learning rate.", 8),
    "dataset": (table_dataset, "Fold sizes after curation; photographs of the same specimen are kept within one fold.", 8.5),
    "architectures": (table_architectures, "GMAC: billions of multiply–accumulate operations for one 224 × 224 image. Activation sites: hidden activation modules replaced in each experiment (DenseNet-121 includes its final activation).", 8.5),
    "augmentation": (table_augmentation, "Augmentation is applied on the fly, so every epoch presents a newly transformed view of each training image; the number of views therefore depends on the number of epochs a run lasted (early stopping).", 8.5),
    "system": (table_system, None, 8.5),
    "stats": (table_stats, "Δ: mean ± SD paired difference in macro-F1 (percentage points) relative to the native activation. Families: piecewise-linear (ReLU, ReLU6, Leaky ReLU, PReLU) and smooth (Swish, Mish, GELU).", 8),
    "perclass": (table_perclass, None, 8),
    "ablation": (table_ablation, "Δ: difference to the full protocol in percentage points.", 8.5),
    "compute": (table_compute, "Latency: single image (batch size 1), mean of 100 GPU or 20 CPU runs after warm-up. EfficientNet-B3 was evaluated at 224 × 224 pixels like all other networks. Per-configuration values, including all activations, are given in Supplementary Table S3.", 8),
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

CITATION = re.compile(r"\[(\d+(?:\s*[–,-]\s*\d+)*)\]")
NEW_OF = {}   # old -> new citation number, filled by renumber_citations and applied to table cells


def remap(text: str) -> str:
    if not NEW_OF:
        return text
    return CITATION.sub(lambda m: "[" + _compress([NEW_OF[n] for n in _expand(m.group(1))]) + "]", text)


def _expand(group: str) -> list:
    nums = []
    for part in group.split(","):
        part = part.strip()
        if re.search(r"[–-]", part):
            a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
            nums.extend(range(a, b + 1))
        else:
            nums.append(int(part))
    return nums


def _compress(nums: list) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def renumber_citations(text: str, refs: list):
    """Renumber citations in order of first appearance (Vancouver style) and reorder the list to match."""
    order = []
    for m in CITATION.finditer(text):
        for n in _expand(m.group(1)):
            if n not in order:
                order.append(n)
    missing = sorted(set(range(1, len(refs) + 1)) - set(order))
    if missing:
        raise SystemExit(f"References never cited: {missing}")
    new_of = {old: new for new, old in enumerate(order, 1)}
    NEW_OF.update(new_of)
    text = CITATION.sub(lambda m: "[" + _compress([new_of[n] for n in _expand(m.group(1))]) + "]", text)
    return text, [refs[old - 1] for old in order]

def build_manuscript(gradcam_sentence: str) -> str:
    doc = Document()
    setup(doc)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(15)
    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, (name, sup, _) in enumerate(AUTHORS):
        a.add_run(name)
        s = a.add_run(sup)
        s.font.superscript = True
        if i < len(AUTHORS) - 1:
            a.add_run(", ")
    for name, sup, aff in AUTHORS:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        s = p.add_run(sup)
        s.font.superscript = True
        p.add_run(" " + aff).font.size = Pt(10)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.0
    p.add_run("Corresponding author: ").bold = True
    p.add_run("Serhat Kılıçarslan (skilicarslan@bandirma.edu.tr)").font.size = Pt(10)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT)
    p = doc.add_paragraph()
    p.add_run("Keywords: ").bold = True
    p.add_run(KEYWORDS)

    table_no, fig_no = 0, 0
    h1, h2 = 0, 0
    text = open(os.path.join(MS, "part1_intro_methods.md"), encoding="utf-8").read() + "\n\n" + \
        open(os.path.join(MS, "part2_results_discussion.md"), encoding="utf-8").read()
    text, references = renumber_citations(text, REFERENCES)
    text = text.replace("[[GRADCAM_SENTENCE]]", gradcam_sentence)
    text = text.replace("[[FUNDING]]", "This research received no external funding.")
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            title = block[2:].strip()
            if title == "Declarations":
                doc.add_heading(title, level=1)
            else:
                h1 += 1
                h2 = 0
                doc.add_heading(f"{h1} {title}", level=1)
        elif block.startswith("## "):
            h2 += 1
            doc.add_heading(f"{h1}.{h2} {block[3:].strip()}", level=2)
        elif block.startswith("[[TABLE:"):
            key = block[8:-2]
            fn, note, font = TABLES[key] if key in TABLES else (None, None, 8)
            table_no += 1
            if key == "main":
                df, bold = table_main()
                add_table(doc, df, TABLE_CAPTIONS[key], table_no,
                          "Rows: architectures; columns: activation functions. ReLU6 is evaluated for MobileNetV2 only, where it is the native activation.",
                          font=7.5, bold_cells=bold)
            else:
                add_table(doc, fn(), TABLE_CAPTIONS[key], table_no, note, font=font)
        elif block.startswith("[[FIGURE:"):
            fig_no += 1
            add_figure(doc, block[9:-2], fig_no)
        elif all(line.startswith("- ") for line in block.splitlines()):
            for line in block.splitlines():
                add_runs(doc.add_paragraph(style="List Bullet"), line[2:].strip())
        elif re.match(r"^\d+\. ", block):
            for line in block.splitlines():
                add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line.strip()))
        else:
            add_runs(doc.add_paragraph(), " ".join(block.splitlines()))

    doc.add_heading("References", level=1)
    for i, ref in enumerate(references, 1):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(4)
        p.add_run(f"{i}. {ref}").font.size = Pt(10)

    path = os.path.join(MS, "Revised_Manuscript_Potato_Tuber_Disease_CNN_Activation.docx")
    doc.save(path)
    print(f"[saved] {path}  ({table_no} tables, {fig_no} figures, {len(REFERENCES)} references)")
    return path


def build_supplement() -> str:
    doc = Document()
    setup(doc)
    for sec in doc.sections:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    t = doc.add_paragraph()
    r = t.add_run("Supplementary material\n" + TITLE)
    r.bold = True

    s = pd.read_csv(os.path.join(TAB, "all_configurations.csv"))
    s["order"] = s.arch.map(C.ARCHITECTURES.index)
    s = s.sort_values(["order", "act"])
    rows = []
    for _, r in s.iterrows():
        rows.append([ARCH[r.arch], ACT[r.act] + ("*" if r.native else ""), pm(r.accuracy_mean, r.accuracy_std),
                     pm(r.error_rate_mean, r.error_rate_std), pm(r.precision_macro_mean, r.precision_macro_std),
                     pm(r.recall_macro_mean, r.recall_macro_std), pm(r.specificity_macro_mean, r.specificity_macro_std),
                     pm(r.f1_macro_mean, r.f1_macro_std), f"{r.auc_macro_mean:.3f} ± {r.auc_macro_std:.3f}",
                     f"{100 * r.pooled_accuracy:.1f} [{100 * r.acc_ci_lo:.1f}, {100 * r.acc_ci_hi:.1f}]",
                     f"{r.epochs_run_mean:.0f} / {r.best_epoch_mean:.0f}", r.lr_counts])
    add_table(doc, pd.DataFrame(rows, columns=["Architecture", "Activation", "Accuracy", "Error rate", "Precision",
                                               "Recall (bal. acc.)", "Specificity", "Macro-F1", "Macro-AUC",
                                               "Pooled accuracy [95% CI]", "Epochs run / best epoch", "Selected LRs (×rotations)"]),
              "All 49 configurations: test metrics (%) as mean ± SD over five rotations (macro-averaged over classes), pooled out-of-fold accuracy with bootstrap 95% CI, mean training length and the learning rates selected on the validation folds (* native activation).",
              "S1", font=6.5)

    lr = pd.read_csv(os.path.join(TAB, "selected_lr_counts.csv"))
    lr["arch"] = lr["arch"].map(ARCH)
    lr.columns = ["Architecture"] + [f"LR {float(c):.0e}" for c in lr.columns[1:]]
    add_table(doc, lr, "Number of rotation × activation combinations in which each learning rate was selected on the validation fold. Grids: {3e-5, 1e-4, 3e-4, 1e-3} for batch-normalised networks, {1e-5, 3e-5, 1e-4, 3e-4} for VGG.", "S2", font=8)

    b = benchmark()
    b["order"] = b.arch.map(C.ARCHITECTURES.index)
    b = b.sort_values(["order", "act"])
    rows = [[ARCH[r.arch], ACT[r.act] + ("*" if r.native else ""), f"{r.params_m:.3f}", f"{r.gmacs:.3f}",
             f"{r.gpu_ms:.2f} ± {r.gpu_ms_std:.2f}", f"{r.cpu_ms:.1f} ± {r.cpu_ms_std:.1f}", f"{r.gpu_img_per_s_b32:.0f}"]
            for _, r in b.iterrows()]
    add_table(doc, pd.DataFrame(rows, columns=["Architecture", "Activation", "Params (M)", "GMAC", "GPU latency (ms)",
                                               "CPU latency (ms)", "GPU throughput (img/s)"]),
              "Inference cost of every configuration on an RTX 3050 Ti laptop GPU and an Intel Core i5-11320H CPU (batch size 1 for latency, 32 for throughput).", "S3", font=7)

    sw = pd.read_csv(os.path.join(TAB, "stats_swap_vs_native.csv"))
    rows = [[r.arch, r.native, r.swap, f"{r.delta_f1_pp:+.1f} ± {r.delta_sd_pp:.1f}", f"{r.wins}/{r.folds}",
             f"{r.p_wilcoxon:.3f}", f"{r.p_holm:.3f}"] for r in sw.itertuples()]
    add_table(doc, pd.DataFrame(rows, columns=["Architecture", "Native", "Replacement", "Δ macro-F1 (pp)", "Rotations improved",
                                               "p (Wilcoxon)", "p (Holm)"]),
              "Paired comparison of every replacement with the native activation over the five rotations.", "S4", font=8)

    path = os.path.join(MS, "Supplementary_Material.docx")
    doc.save(path)
    print(f"[saved] {path}")
    return path


def build_response(fill: dict) -> str:
    text = open(os.path.join(MS, "response_to_reviewers.md"), encoding="utf-8").read()
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    if left:
        raise SystemExit(f"Unfilled placeholders in the response letter: {sorted(set(left))}")
    with open(os.path.join(MS, "response_to_reviewers_filled.md"), "w", encoding="utf-8") as f:
        f.write(text)
    doc = Document()
    setup(doc)
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block == "---":
            continue
        if block.startswith("# "):
            doc.add_heading(block[2:], level=1)
        elif block.startswith("## "):
            doc.add_heading(block[3:], level=2)
        elif re.match(r"^\d+\. ", block):
            for line in block.splitlines():
                add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line.strip()))
        else:
            add_runs(doc.add_paragraph(), " ".join(block.splitlines()))
    path = os.path.join(MS, "Response_to_Reviewers.docx")
    doc.save(path)
    print(f"[saved] {path}")
    return path


if __name__ == "__main__":
    fill_path = os.path.join(MS, "response_fill.json")
    fill = json.load(open(fill_path, encoding="utf-8")) if os.path.exists(fill_path) else {}
    build_manuscript(fill.get("GRADCAM_SENTENCE_MS", ""))
    build_supplement()
    if fill:
        build_response(fill)
