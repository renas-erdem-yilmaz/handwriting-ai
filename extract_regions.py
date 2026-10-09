import cv2
import numpy as np
from pathlib import Path

INPUT_FOLDER = Path("data/pages")
OUTPUT_FOLDER = Path("data/crops")
DEBUG_FOLDER = Path("data/debug")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

for file in OUTPUT_FOLDER.glob("*.png"):
    file.unlink()

for file in DEBUG_FOLDER.glob("*.png"):
    file.unlink()


def split_region_into_lines(binary, x, y, w, h):

    roi = binary[y:y + h, x:x + w]

    row_ink = np.sum(
        roi > 0,
        axis=1
    ).astype(np.float32)

    window_size = 7

    kernel = np.ones(window_size) / window_size

    smoothed = np.convolve(
        row_ink,
        kernel,
        mode="same"
    )

    threshold = max(
        5,
        w * 0.006
    )

    active_rows = smoothed > threshold

    bands = []

    start = None

    for i, active in enumerate(active_rows):

        if active and start is None:
            start = i

        elif not active and start is not None:

            end = i

            if end - start >= 12:
                bands.append((start, end))

            start = None

    if start is not None:

        end = len(active_rows)

        if end - start >= 12:
            bands.append((start, end))

    merged = []

    for band in bands:

        if not merged:
            merged.append(list(band))
            continue

        previous = merged[-1]

        gap = band[0] - previous[1]

        if gap < 10:
            previous[1] = band[1]

        else:
            merged.append(list(band))

    result = []

    for start_y, end_y in merged:

        line_y = y + start_y
        line_h = end_y - start_y

        if line_h < 15:
            continue

        result.append(
            (
                x,
                line_y,
                w,
                line_h
            )
        )

    if len(result) == 0:
        return [(x, y, w, h)]

    return result


page_files = sorted(
    INPUT_FOLDER.glob("*.png")
)

crop_id = 1


for page_path in page_files:

    print(f"Processing {page_path.name}")

    image = cv2.imread(
        str(page_path)
    )

    if image is None:
        print(
            f"Could not read {page_path.name}"
        )
        continue

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    left_ignore = int(
        width * 0.055
    )

    right_ignore = int(
        width * 0.055
    )

    binary[
        :,
        :left_ignore
    ] = 0

    binary[
        :,
        width - right_ignore:
    ] = 0

    top_ignore = int(
        height * 0.025
    )

    bottom_ignore = int(
        height * 0.025
    )

    binary[
        :top_ignore,
        :
    ] = 0

    binary[
        height - bottom_ignore:,
        :
    ] = 0

    horizontal_size = max(
        45,
        int(width * 0.04)
    )

    vertical_size = max(
        2,
        int(height * 0.0015)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            horizontal_size,
            vertical_size
        )
    )

    connected = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    initial_regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 100:
            continue

        if h < 18:
            continue

        if w > width * 0.97:
            continue

        if (
            w > width * 0.50
            and h < 25
        ):
            continue

        if y > height * 0.94:
            continue

        initial_regions.append(
            (x, y, w, h)
        )

    final_regions = []

    normal_line_height = (
        height * 0.035
    )

    for x, y, w, h in initial_regions:

        if h > normal_line_height:

            split_regions = (
                split_region_into_lines(
                    binary,
                    x,
                    y,
                    w,
                    h
                )
            )

            final_regions.extend(
                split_regions
            )

        else:

            final_regions.append(
                (x, y, w, h)
            )

    final_regions.sort(
        key=lambda box: (
            box[1],
            box[0]
        )
    )

    debug_image = image.copy()

    for x, y, w, h in final_regions:

        cv2.rectangle(
            debug_image,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

    debug_path = (
        DEBUG_FOLDER
        / page_path.name
    )

    cv2.imwrite(
        str(debug_path),
        debug_image
    )

    for region_number, (
        x,
        y,
        w,
        h
    ) in enumerate(
        final_regions,
        start=1
    ):

        padding_x = 20
        padding_y = 12

        x1 = max(
            0,
            x - padding_x
        )

        y1 = max(
            0,
            y - padding_y
        )

        x2 = min(
            width,
            x + w + padding_x
        )

        y2 = min(
            height,
            y + h + padding_y
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        filename = (
            f"{page_path.stem}"
            f"_region_{region_number:03d}"
            f"_crop_{crop_id:04d}.png"
        )

        cv2.imwrite(
            str(
                OUTPUT_FOLDER
                / filename
            ),
            crop
        )

        crop_id += 1

    print(
        f"  Initial regions: "
        f"{len(initial_regions)}"
    )

    print(
        f"  Final line regions: "
        f"{len(final_regions)}"
    )


print()

print("--------------------------------")
print("STEP 2 PROCESSING COMPLETE")
print("--------------------------------")

print(
    f"Created {crop_id - 1} crops."
)

print(
    f"Crops saved to: {OUTPUT_FOLDER}"
)

print(
    f"Debug images saved to: {DEBUG_FOLDER}"
)