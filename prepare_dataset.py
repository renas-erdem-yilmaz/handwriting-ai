import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CROP_FOLDER = Path("data/crops")
LABELS_FILE = Path("data/labels.csv")
OUTPUT_FOLDER = Path("data/dataset")

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)

if OUTPUT_FOLDER.exists():
    shutil.rmtree(OUTPUT_FOLDER)

for split in ["train", "val", "test"]:
    (OUTPUT_FOLDER / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

samples = []

with open(
    LABELS_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        filename = row["filename"].strip()
        text = row["text"].strip()
        content_type = row["type"].strip()
        keep = row["keep"].strip().lower()

        if keep != "yes":
            continue

        if content_type == "diagram":
            continue

        if not text:
            continue

        image_path = CROP_FOLDER / filename

        if not image_path.exists():
            print(f"Missing image: {filename}")
            continue

        match = re.match(
            r"(page_\d+)",
            filename
        )

        if match:
            page = match.group(1)
        else:
            page = "unknown"

        samples.append({
            "filename": filename,
            "text": text,
            "type": content_type,
            "page": page
        })

print()
print(f"Usable labeled samples: {len(samples)}")

if len(samples) < 10:
    raise RuntimeError(
        "Not enough usable samples."
    )

pages = defaultdict(list)

for sample in samples:
    pages[sample["page"]].append(sample)

page_names = list(pages.keys())

random.shuffle(page_names)

splits = {
    "train": [],
    "val": [],
    "test": []
}

if len(page_names) >= 3:

    target_counts = {
        "train": len(samples) * TRAIN_RATIO,
        "val": len(samples) * VAL_RATIO,
        "test": len(samples) * TEST_RATIO
    }

    first_page = page_names.pop()
    second_page = page_names.pop()
    third_page = page_names.pop()

    splits["train"].extend(
        pages[first_page]
    )

    splits["val"].extend(
        pages[second_page]
    )

    splits["test"].extend(
        pages[third_page]
    )

    for page in page_names:

        deficits = {}

        for split_name in splits:

            deficits[split_name] = (
                target_counts[split_name]
                - len(splits[split_name])
            )

        destination = max(
            deficits,
            key=deficits.get
        )

        splits[destination].extend(
            pages[page]
        )

else:

    print()
    print(
        "WARNING: Labels come from fewer than "
        "3 different pages."
    )

    print(
        "Falling back to crop-level splitting."
    )

    random.shuffle(samples)

    train_end = int(
        len(samples) * TRAIN_RATIO
    )

    val_end = train_end + int(
        len(samples) * VAL_RATIO
    )

    splits["train"] = samples[:train_end]

    splits["val"] = samples[
        train_end:val_end
    ]

    splits["test"] = samples[
        val_end:
    ]


def save_split(
    split_name,
    split_samples
):

    csv_path = (
        OUTPUT_FOLDER
        / split_name
        / "labels.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "text",
                "type",
                "page"
            ]
        )

        writer.writeheader()

        for sample in split_samples:

            writer.writerow(sample)

            source = (
                CROP_FOLDER
                / sample["filename"]
            )

            destination = (
                OUTPUT_FOLDER
                / split_name
                / "images"
                / sample["filename"]
            )

            shutil.copy2(
                source,
                destination
            )


for split_name in splits:

    save_split(
        split_name,
        splits[split_name]
    )


all_characters = sorted(
    set(
        "".join(
            sample["text"]
            for sample in samples
        )
    )
)

charset_path = (
    OUTPUT_FOLDER
    / "charset.txt"
)

with open(
    charset_path,
    "w",
    encoding="utf-8"
) as f:

    for character in all_characters:

        if character == " ":
            display = "<SPACE>"

        elif character == "\n":
            display = "<NEWLINE>"

        else:
            display = character

        f.write(display + "\n")


type_counts = Counter(
    sample["type"]
    for sample in samples
)

print()
print("--------------------------------")
print("DATASET SUMMARY")
print("--------------------------------")

for split_name in [
    "train",
    "val",
    "test"
]:

    print(
        f"{split_name.upper():5}: "
        f"{len(splits[split_name])}"
    )

print()

print("Sample types:")

for sample_type, count in type_counts.items():

    print(
        f"  {sample_type}: {count}"
    )

print()

print(
    f"Unique characters: "
    f"{len(all_characters)}"
)

print()

print(
    "Characters:"
)

print(
    " ".join(
        char if char != " "
        else "<SPACE>"
        for char in all_characters
    )
)

print()

print(
    f"Dataset created at: "
    f"{OUTPUT_FOLDER}"
)