from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def main() -> None:
    output_path = Path("assignment/data/corpus/acme_data_residency.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text_lines = [
        "AcmeOps Data Residency",
        "",
        "AcmeOps stores production customer data in the US region by default.",
        "Enterprise customers can request EU regional storage during onboarding.",
        "Data residency changes require a planned migration window of at least",
        "two business days.",
        "",
        "Backups are encrypted and retained for 35 days. Audit log exports are",
        "not moved automatically during a residency migration; customers must",
        "export historical logs before the migration if they need an archive.",
        "",
        "Metadata: product=acmeops, doc_type=compliance, plan=enterprise",
    ]

    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    y = height - 72
    for line in text_lines:
        pdf.drawString(72, y, line)
        y -= 16
    pdf.save()
    print(output_path)


if __name__ == "__main__":
    main()
