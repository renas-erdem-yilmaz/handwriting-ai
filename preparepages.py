import fitz
from pathlib import Path

PDF_FOLDER = Path("data/pdfs")
OUTPUT_FOLDER = Path("data/pages")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

page_number = 1

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path.name}")

    document = fitz.open(pdf_path)

    for page in document:
        pix = page.get_pixmap(dpi=300, alpha=False)

        filename = f"page_{page_number:03d}.png"
        output_path = OUTPUT_FOLDER / filename

        pix.save(output_path)

        print(f"  Saved {filename}")

        page_number += 1

print()
print(f"Finished. Exported {page_number - 1} pages.")