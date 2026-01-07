import os
import cv2
import csv
from glob import glob


# --------------------------------------------------
# CSV UTIL
# --------------------------------------------------

def append_rating(csv_path, filename, rating):
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["filename", "rating"])

        writer.writerow([filename, rating])
        print([filename, rating])


# --------------------------------------------------
# DISPLAY + INPUT
# --------------------------------------------------

def show_and_rate(image_path):
    """
    Displays image and waits for user input.
    Returns:
        "1", "2", "3" → rating
        "0"           → skip
        None          → quit
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return "0"

    title = os.path.basename(image_path)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1400, 500)

    instructions = "1=Bad  2=OK  3=Good  0=Skip  q=Quit"

    cv2.putText(
        img,
        instructions,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(title, img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("1"), ord("2"), ord("3"), ord("0")):
            cv2.destroyWindow(title)
            return chr(key)

        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

def evaluate_folder(
    image_folder,
    csv_path=None,
    extensions=("*.png", "*.jpg", "*.jpeg")
):
    if csv_path is None:
        csv_path = os.path.join(image_folder, "ratings.csv")

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(image_folder, ext)))

    image_paths = sorted(image_paths)

    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images")
    print("Press 1/2/3 to rate, 0 to skip, q to quit")

    for img_path in image_paths:
        rating = show_and_rate(img_path)

        if rating is None:
            print("Evaluation stopped by user.")
            break

        if rating != "0":
            append_rating(
                csv_path,
                os.path.basename(img_path),
                rating
            )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    evaluate_folder(
        image_folder="./facsimile/results/test120_results",
        csv_path="./facsimile/results/ratings120.csv"
    )
