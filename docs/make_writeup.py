"""Generate a short PDF report from the locally generated experiment results."""
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "model_comparison.csv"
OUT = ROOT / "docs" / "writeup.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9, leading=11)
small = ParagraphStyle("small", parent=body, fontSize=8)
head = ParagraphStyle("head", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)

def P(x, style=body): return Paragraph(x, style)

def build():
    results = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm)
    story = [P("Airbnb Price Prediction", ss["Title"]), P("Dataset-driven regression experiment", small), Spacer(1, 5)]
    story += [P("1. Objective", head), P("Predict the supplied dataset's log-transformed Airbnb listing price from listing characteristics, location, reviews, amenities and textual information." )]
    story += [P("2. Dataset", head), P("The experiment uses 74,111 listings from NYC, LA, SF, DC, Chicago and Boston. The target is the dataset's existing <b>log_price</b> field. The data are split 80:20 using city-stratified sampling." )]
    story += [P("3. Method", head), P("Numeric variables are median-imputed and standardized. Categorical variables are one-hot encoded. Amenities and listing text are represented with TF-IDF. All preprocessing is fitted only on the training partition to prevent test leakage. Ridge regression, Random Forest and XGBoost are evaluated on the held-out test set." )]
    story += [P("4. Results", head)]
    if results.empty:
        story += [P("Run <font face='Courier'>python run_experiments.py</font> to generate the numerical results.")]
    else:
        rows = [["Model", "MAE", "RMSE", "R²", "Time (s)"]]
        for _, r in results.iterrows():
            rows.append([r.model, f"{r.mae_log_price:.4f}", f"{r.rmse_log_price:.4f}", f"{r.r2:.4f}", f"{r.seconds:.2f}"])
        t = Table(rows, colWidths=[35*mm, 30*mm, 30*mm, 25*mm, 30*mm])
        t.setStyle(TableStyle([("GRID", (0,0),(-1,-1),0.4,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.lightgrey), ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("ALIGN",(1,1),(-1,-1),"CENTER")]))
        story += [t]
    story += [P("5. Conclusion", head), P("The reported metrics are produced directly from the supplied dataset and the code in this repository.", small)]
    doc.build(story)

if __name__ == "__main__":
    build()
