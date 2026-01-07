import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from .src.segmentation import find_contour, segment_hsv, segment_saturation
from .src.binarization import binarize, all_binarize_algos
from .src.utils import save_image_cv, imshow

def process_directory(folder_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted(
        glob(os.path.join(folder_path, "*.jpg")) +
        glob(os.path.join(folder_path, "*.JPG")) +
        glob(os.path.join(folder_path, "*.png")) +
        glob(os.path.join(folder_path, "*.tiff"))
    )
    process_pipeline(
        uploaded_paths=image_paths,
        options=DEFAULT_OPTIONS,
        output_folder=output_folder,
        testing=True
    )

def overlay_outline(image, binary):
    overlay = image.copy()
    mask = binary == 0
    overlay[mask] = [0, 0, 255]
    return overlay

def save_single_triplet_plot(img, result, overlay, filename, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result, cmap="gray")
    axes[1].set_title("Processed")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        output_folder,
        f"{os.path.splitext(filename)[0]}_triplet.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")

# ------------------------------
# Full pipeline
# ------------------------------
def process_pipeline(uploaded_paths, options, output_folder, 
                     debug=False, testing=False):
    processed_names = []
    for path in uploaded_paths:
        result = process_single(path, options, output_folder)
        # ---- TESTING MODE ONLY ----
        if testing:
            img = cv2.imread(path)
            overlay = overlay_outline(img, result)
            save_single_triplet_plot(
                img=img,
                result=result,
                overlay=overlay,
                filename=os.path.basename(path),
                output_folder=output_folder
            )
        else:
            out_name = os.path.basename(path)
            out_path = os.path.join(output_folder, out_name)
            save_image_cv(result, out_path)
            processed_names.append(out_name)

    return processed_names


def process_single(img_path, options, output_folder, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load image")

    # --- BINARIZATION ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug:
        imshow("gray", gray)
    bin_opts = options["binarize"]
    # bilateral filter preprocess
    bin_bilat = bin_opts["bilateral"]
    if bin_bilat["d"] > 0:
        gray = cv2.bilateralFilter(
            gray,
            d=bin_bilat["d"],
            sigmaColor=bin_bilat["sigma_color"],
            sigmaSpace=bin_bilat["sigma_space"],
        )
    if debug:
        imshow("bilat", gray)
    # binarize
    binary_img = binarize(gray, func=bin_opts["method"])

    # postprocess 
    mk = bin_opts["median_k"] # median filter
    if mk and mk > 1:
        binary_img = cv2.medianBlur(binary_img, mk | 1)
    if debug:
        imshow("with median", binary_img)

    mck = bin_opts["morph_close_k"] # morph close
    if mck and mck > 0:
        kernel = np.ones((mck, mck), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    if debug:
        imshow("morph close", binary_img)

    mok = bin_opts["morph_open_k"] # morph open
    if mok and mok > 0:
        kernel = np.ones((mok, mok), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    if debug:
        imshow("morph open", binary_img)
    

    # --- SEGMENTATION ---
    seg_opts = options["segment"]
    # bilateral filter preprocess
    seg_bilat = seg_opts["bilateral"]
    if seg_bilat["d"] > 0:
        gray = cv2.bilateralFilter(
            gray,
            d=seg_bilat["d"],
            sigmaColor=seg_bilat["sigma_color"],
            sigmaSpace=seg_bilat["sigma_space"],
        )
    # segment
    if seg_opts["method"] == "kmeans":
        mask = segment_hsv(img)
    elif seg_opts["method"] == "saturation":
        mask = segment_saturation(img)
    else:
        raise ValueError("invalid segmentation method")
    # postprocess
    mck = seg_opts["morph_close_k"] # morph close
    if mck and mck > 0:
        kernel = np.ones((mck, mck), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # find contour line of biggest mask part
    contour_line, mask = find_contour(mask)

    # --- COMPOSITE OUTPUT ---
    h, w = mask.shape
    output = np.full((h, w), 255, dtype=np.uint8)
    output[mask == 255] = binary_img[mask == 255]
    contour_pixels = contour_line > 0
    output[contour_pixels] = 0

    final = output.copy()

    if contour_line is not None and options.get("overlay", True):
        final = overlay_outline(img, output)

    return final


DEFAULT_OPTIONS = {
    "binarize": {
        "method": "WOLF",
        "bilateral": {"d": 5, "sigma_color": 1e6, "sigma_space": 20},
        "median_k": 5,
        "morph_close_k": 1,
        "morph_open_k": 5,
    },
    "segment": {
        "method": "saturation",
        "bilateral": {"d": 5, "sigma_color": 50, "sigma_space": 20},
        "morph_close_k": 20,
    },
    "overlay": False,
}


if __name__ == "__main__":
    # must be run from outside of facsimile folder
     process_directory("./facsimile/data/test120_imgs", output_folder="./facsimile/results/test120_results")
    # process_directory("./image_stitching/data/mma_krater", output_folder="./facsimile/results/krater_fac")
    # process_single("./data/original-1-7.JPG", method="color", debug=True)
    # img_file = "./image_stitching/data/mma_krater/Screenshot 2025-12-11 at 5.25.52 PM.png"
    # process_single(img_file, DEFAULT_OPTIONS, "./facsimile/results", debug=True)
