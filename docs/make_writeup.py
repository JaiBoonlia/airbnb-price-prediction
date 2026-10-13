
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "docs" / "writeup.pdf"

MODEL_RESULTS = [
    ["Model", "MAE", "RMSE", "R²"],
    ["Linear Regression", "0.3056", "0.4187", "0.6588"],
    ["Random Forest", "0.2830", "0.3952", "0.6960"],
    ["XGBoost", "0.2843", "0.3911", "0.7022"],
]


def build_report():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Airbnb Price Prediction",
        author="Airbnb Price Prediction Project",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#234E70"),
            spaceAfter=12,
        )
    )

    story = []

    story.append(Paragraph("Airbnb Price Prediction", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Model Evaluation and Results",
            styles["Heading2"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Problem Statement", styles["Heading2"]))
    story.append(
        Paragraph(
            "This project investigates whether machine learning models can "
            "estimate Airbnb listing prices from listing characteristics and "
            "location information across six US cities.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Dataset and Methodology", styles["Heading2"]))
    story.append(
        Paragraph(
            "The dataset contains 74,111 listing records. The target variable "
            "is log_price. The project compares Linear Regression, Random "
            "Forest, and XGBoost using MAE, RMSE, and R² on a held-out test set.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Model Performance", styles["Heading2"]))

    table = Table(
        MODEL_RESULTS,
        colWidths=[6.0 * cm, 2.5 * cm, 2.5 * cm, 2.0 * cm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234E70")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#EEF3F7")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "MAE and RMSE are reported on the log-price scale; they are not "
            "direct dollar-error measurements.",
            styles["Italic"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Results and Interpretation", styles["Heading2"]))
    story.append(
        Paragraph(
            "XGBoost achieved the highest R² (0.7022) and lowest RMSE (0.3911). "
            "Random Forest achieved the lowest MAE (0.2830). XGBoost therefore "
            "performed best on two of the three reported metrics, but it did "
            "not achieve the lowest error on every metric.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Limitations", styles["Heading2"]))
    story.append(
        Paragraph(
            "The dataset covers six US cities and may not generalise to other "
            "markets. Prediction quality depends on the available features, "
            "data quality, and factors not represented in the dataset. Feature "
            "importance describes predictive contribution and does not prove "
            "causation.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Conclusion", styles["Heading2"]))
    story.append(
        Paragraph(
            "The experiments show that the evaluated regression models capture "
            "useful patterns in Airbnb listing prices. XGBoost offers the "
            "strongest overall performance by R² and RMSE, while Random Forest "
            "has a slightly lower MAE. Predictions should be treated as "
            "estimates rather than guaranteed listing prices.",
            styles["BodyText"],
        )
    )

    document.build(story)
    print(f"Report generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_report()
