import csv
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

CROP_FOLDER = Path("data/crops")
CSV_PATH = Path("data/labels.csv")

crop_files = sorted(CROP_FOLDER.glob("*.png"))

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
print("Loading handwriting OCR model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(device)

model.eval()

print("Model loaded.")

existing = {}

if CSV_PATH.exists():
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            existing[row["filename"]] = row


def save_csv():
    rows = []

    for crop in crop_files:
        if crop.name in existing:
            rows.append(existing[crop.name])

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "text",
                "type",
                "keep"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def recognize_text(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()

    except Exception as e:
        print(f"OCR error: {e}")
        return ""


class AnnotationApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Handwriting Dataset Annotator")
        self.root.geometry("1300x820")

        self.index = 0
        self.photo = None
        self.last_ocr_file = None

        self.find_first_unlabeled()

        self.counter_label = tk.Label(
            root,
            font=("Arial", 14)
        )

        self.counter_label.pack(pady=10)

        self.image_frame = tk.Frame(root)
        self.image_frame.pack(
            fill="both",
            expand=True
        )

        self.image_label = tk.Label(
            self.image_frame
        )

        self.image_label.pack(
            expand=True
        )

        form = tk.Frame(root)

        form.pack(
            fill="x",
            padx=30,
            pady=10
        )

        tk.Label(
            form,
            text="Transcription:",
            font=("Arial", 12)
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.text_entry = tk.Entry(
            form,
            font=("Arial", 15)
        )

        self.text_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=10
        )

        form.columnconfigure(
            1,
            weight=1
        )

        tk.Label(
            form,
            text="Type:",
            font=("Arial", 12)
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=10
        )

        self.type_var = tk.StringVar(
            value="text"
        )

        self.type_menu = ttk.Combobox(
            form,
            textvariable=self.type_var,
            state="readonly",
            values=[
                "text",
                "heading",
                "equation",
                "mixed",
                "diagram"
            ]
        )

        self.type_menu.grid(
            row=1,
            column=1,
            sticky="w",
            padx=10
        )

        symbol_frame = tk.Frame(root)
        symbol_frame.pack(pady=5)

        symbols = [
            "α", "β", "θ", "φ",
            "⇒", "→", "Δ", "μ",
            "ω", "λ", "π",
            "^2", "^3", "_0"
        ]

        for i, symbol in enumerate(symbols):

            button = tk.Button(
                symbol_frame,
                text=symbol,
                width=5,
                font=("Arial", 12),
                command=lambda s=symbol: self.insert_symbol(s)
            )

            button.grid(
                row=0,
                column=i,
                padx=2
            )

        buttons = tk.Frame(root)

        buttons.pack(
            pady=15
        )

        tk.Button(
            buttons,
            text="← Previous",
            width=15,
            command=self.previous
        ).grid(
            row=0,
            column=0,
            padx=5
        )

        tk.Button(
            buttons,
            text="Run OCR Again",
            width=15,
            command=self.run_ocr
        ).grid(
            row=0,
            column=1,
            padx=5
        )

        tk.Button(
            buttons,
            text="Discard",
            width=15,
            command=self.discard
        ).grid(
            row=0,
            column=2,
            padx=5
        )

        tk.Button(
            buttons,
            text="Save + Next",
            width=15,
            command=self.save_and_next
        ).grid(
            row=0,
            column=3,
            padx=5
        )

        tk.Button(
            buttons,
            text="Skip",
            width=15,
            command=self.skip
        ).grid(
            row=0,
            column=4,
            padx=5
        )

        self.root.bind(
            "<Control-Return>",
            lambda event: self.save_and_next()
        )

        self.root.bind(
            "<Control-BackSpace>",
            lambda event: self.discard()
        )

        self.root.bind(
            "<Left>",
            lambda event: self.previous()
        )

        self.root.bind(
            "<Right>",
            lambda event: self.skip()
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

        self.load_current()


    def find_first_unlabeled(self):

        for i, crop in enumerate(crop_files):

            if crop.name not in existing:
                self.index = i
                return

        self.index = 0


    def load_current(self):

        if not crop_files:
            return

        crop = crop_files[self.index]

        image = Image.open(crop)

        image.thumbnail(
            (1150, 430)
        )

        self.photo = ImageTk.PhotoImage(
            image
        )

        self.image_label.config(
            image=self.photo
        )

        self.counter_label.config(
            text=(
                f"{self.index + 1} / "
                f"{len(crop_files)}   |   "
                f"{crop.name}"
            )
        )

        self.text_entry.delete(
            0,
            tk.END
        )

        self.type_var.set(
            "text"
        )

        if crop.name in existing:

            row = existing[crop.name]

            self.text_entry.insert(
                0,
                row["text"]
            )

            self.type_var.set(
                row["type"]
            )

        else:

            self.root.update()

            self.counter_label.config(
                text=(
                    f"{self.index + 1} / "
                    f"{len(crop_files)}   |   "
                    f"{crop.name}   |   OCR..."
                )
            )

            self.root.update()

            prediction = recognize_text(
                crop
            )

            self.text_entry.insert(
                0,
                prediction
            )

            self.counter_label.config(
                text=(
                    f"{self.index + 1} / "
                    f"{len(crop_files)}   |   "
                    f"{crop.name}"
                )
            )

        self.text_entry.focus_set()

        self.text_entry.icursor(
            tk.END
        )


    def insert_symbol(self, symbol):

        position = self.text_entry.index(
            tk.INSERT
        )

        self.text_entry.insert(
            position,
            symbol
        )

        self.text_entry.focus_set()


    def run_ocr(self):

        crop = crop_files[self.index]

        prediction = recognize_text(
            crop
        )

        self.text_entry.delete(
            0,
            tk.END
        )

        self.text_entry.insert(
            0,
            prediction
        )

        self.text_entry.focus_set()


    def save_current(self, keep):

        crop = crop_files[self.index]

        existing[crop.name] = {
            "filename": crop.name,
            "text": self.text_entry.get().strip(),
            "type": self.type_var.get(),
            "keep": keep
        }

        save_csv()


    def save_and_next(self):

        text = self.text_entry.get().strip()

        if (
            not text
            and self.type_var.get() != "diagram"
        ):
            return

        self.save_current(
            "yes"
        )

        self.next()


    def discard(self):

        self.save_current(
            "no"
        )

        self.next()


    def skip(self):

        self.next()


    def next(self):

        if self.index < len(crop_files) - 1:

            self.index += 1
            self.load_current()

        else:

            self.counter_label.config(
                text="Reached final crop."
            )


    def previous(self):

        if self.index > 0:

            self.index -= 1
            self.load_current()


    def close(self):

        save_csv()
        self.root.destroy()


root = tk.Tk()

app = AnnotationApp(root)

root.mainloop()