"""Builds docs/writeup.pdf (<= 2 pages). Edit TEAM and re-run: python docs/make_writeup.py"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TEAM = "Team: Jai Boonlia (PES1UG24CS197) and Kanishka Sarraf (PES1UG24CS215) | Section D | Problem 71 | UE24CS352A Machine Learning"
OUT = "docs/writeup.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.2, spaceAfter=3)
h = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=5, spaceAfter=2)
title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=14, spaceAfter=2)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
bul = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=2)

def P(t, st=body): return Paragraph(t, st)
def B(t): return Paragraph(t, bul, bulletText="\u2022")

def table(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t

s = [P("Predicting Airbnb Listing Price Across Different Cities", title), P(TEAM, small), Spacer(1, 3)]

s += [P("1. Problem Statement", h),
      P("Airbnb hosts set nightly prices empirically, which is hard for new hosts and leaves guests unsure whether a "
        "listing is fairly priced. We build a supervised regression system that predicts the nightly listing price "
        "from listing attributes (room size, fees, reviews, location, amenities, text descriptions, dates) for New "
        "York City, Paris and Berlin. We also ask whether a model trained on several cities generalises to a city "
        "it has never seen. The work follows Luo, Zhou and Zhou (2019).")]

s += [P("2. Dataset", h),
      P("Three Kaggle Airbnb listing tables (96 raw columns each): NYC (44,317 listings), Paris (59,881) and Berlin "
        "(22,552). Each city is split train:validation:test = 7:2:1. The target is the listing price in dollars per "
        "night. Features are grouped as continuous (28 after removing highly correlated ones), categorical (20 "
        "encoded features), text (12 columns such as summary, transit, neighbourhood overview) and date "
        "(host_since, first_review, last_review). Identifier-like or near-constant columns (id, host_id, market, "
        "country_code, license) were dropped.")]

s += [P("3. Approach", h),
      B("<b>Outliers / label:</b> either cap price at $500 (removes about 1% of listings) or transform the label; "
        "square-root and logarithm both help, log is best."),
      B("<b>Continuous:</b> 0-fill for security deposit and cleaning fee, mean-fill otherwise, standardise to N(0,1); "
        "optional degree-2 interaction terms."),
      B("<b>Categorical:</b> one-hot encoding; list fields (amenities, host_verifications) turned into multi-hot "
        "vectors through a dictionary."),
      B("<b>Text:</b> tf-idf on unigrams and bigrams, then truncated SVD to 50 dimensions per column (600 in total)."),
      B("<b>Date:</b> null replaced by mean date, converted to days since the earliest date."),
      B("<b>Models:</b> L2-regularised linear regression and KNN (baselines), Random Forest, XGBoost and a "
        "multilayer perceptron. Metrics are R<super>2</super> and MSE."),
      B("<b>Cross-city learning:</b> train the network on NYC, Paris, and NYC+Paris, then test on Berlin, which is "
        "never seen during training.")]

s += [P("4. Implementation Overview", h),
      P("Python repository: <font face='Courier'>config.py</font> holds all hyper-parameters; "
        "<font face='Courier'>src/data.py</font> loads and cleans the CSVs, applies the $500 cap, splits the data and "
        "implements label transforms; <font face='Courier'>src/features.py</font> builds the C / I / O / T / D "
        "feature groups, fitted on the training split only to avoid leakage; <font face='Courier'>src/models.py</font> "
        "wraps scikit-learn and XGBoost models and a PyTorch MLP; <font face='Courier'>run_experiments.py</font> "
        "regenerates every table (<font face='Courier'>--table 1..7</font>) into <font face='Courier'>results/</font>. "
        "The final network has two hidden layers (64, 32) with ReLU, MSE loss, SGD with Nesterov momentum "
        "(lr 0.005, momentum 0.9), L2 = 0.005, and early stopping on a 10% hold-out of the training data "
        "(change in validation R<super>2</super> below 1e-4 for 10 iterations). The smaller network and early "
        "stopping were adopted because the larger 3-hidden-layer model (128, 512, 64) overfitted.")]

s += [P("5. Results", h),
      P("Reference values reported by Luo et al. (2019); full tables are in <font face='Courier'>docs/PAPER_RESULTS.md</font> "
        "and our own runs are written to <font face='Courier'>results/</font>.", small)]
t1 = [["Model (NYC, continuous, price <= $500)", "Train R2", "Test R2", "Test MSE"],
      ["Linear regression (L2)", "0.489", "0.497", "3541.99"], ["K-nearest neighbour", "0.351", "0.323", "4760.61"],
      ["Random forest", "0.752", "0.655", "2427.05"], ["Neural network", "0.702", "0.665", "2367.44"],
      ["XGBoost", "0.669", "0.665", "2357.83"]]
t2 = [["XGBoost configuration", "Train R2", "Test R2"],
      ["C, raw price (no cap)", "0.558", "0.195"], ["C, sqrt label", "0.652", "0.537"], ["C, log label", "0.695", "0.669"],
      ["C + interactions, log label", "0.708", "0.676"], ["Date only (TH)", "0.033", "0.016"],
      ["Categorical only (TH)", "0.546", "0.532"], ["C+O+T (TH)", "0.724", "0.706"], ["C+O+T, log label", "0.749", "0.740"]]
t3 = [["Neural network (C+O, log)", "Train R2", "Test R2"],
      ["NYC", "0.769", "0.741 (NYC)"], ["Paris", "0.762", "0.716 (Paris)"], ["NYC + Paris", "0.816", "0.773 (combined)"]]
t4 = [["Trained on -> tested on Berlin", "Test MSE", "Test R2"],
      ["NYC only", "1.78007", "-3.512"], ["Paris only", "7.26637", "-17.42"], ["NYC + Paris", "0.19163", "0.514"]]
s += [table(t1, [75*mm, 25*mm, 25*mm, 28*mm]), Spacer(1, 4), table(t2, [75*mm, 25*mm, 25*mm]), Spacer(1, 4),
      table(t3, [75*mm, 25*mm, 35*mm]), Spacer(1, 4), table(t4, [75*mm, 25*mm, 25*mm]), Spacer(1, 4)]

s += [P("6. Conclusions", h),
      B("XGBoost and the neural network clearly beat the linear and KNN baselines (test R<super>2</super> about 0.665 "
        "on continuous features alone, versus 0.50 and 0.32)."),
      B("Prices are heavily right-skewed: a log transform raised XGBoost test R<super>2</super> from 0.195 to 0.669 "
        "even without removing outliers."),
      B("Categorical and text features add substantial signal (0.665 to 0.706 / 0.740 with log label), while date "
        "features are nearly uninformative (R<super>2</super> about 0.02)."),
      B("Training on NYC + Paris gave the best network (test R<super>2</super> 0.773) and was the only one that "
        "transferred to Berlin (0.514); single-city models fail badly (negative R<super>2</super>), suggesting "
        "multi-city training keeps generic price drivers and suppresses city-specific ones."),
      B("Limitations: the network overfits when the feature count is large, hyper-parameters were barely tuned, and "
        "Berlin performance (0.514) is still well below in-city performance.")]

s += [P("Reference: Y. Luo, X. Zhou, Y. Zhou, <i>Predicting Airbnb Listing Price Across Different Cities</i>, 2019.", small)]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=13*mm, bottomMargin=12*mm,
                        title="Predicting Airbnb Listing Price Across Different Cities")
doc.build(s)
