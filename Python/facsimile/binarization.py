
from skimage import data
from matplotlib import pyplot as plt
import skimage
import cv2
import numpy as np
from .segmentation import segment_saturation, find_contour

# https://pypi.org/project/doxapy/
import doxapy

algorithms = ['OTSU', 'BERNSEN', 'NIBLACK', 'SAUVOLA', 'WOLF', 'GATOS', 'NICK', 'SU', 'TRSINGH', 'BATAINEH', 'ISAUVOLA', 'WAN']

def all_binarize_algos(gray_img: np.ndarray):
    all_bins = []
    for algo in algorithms:
        binary_image = np.empty(gray_img.shape, gray_img.dtype)
        # Pick an algorithm from the DoxaPy library and convert the image to binary
        facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, algo))
        facsimile.initialize(gray_img)
        facsimile.to_binary(binary_image)
        all_bins.append(binary_image)
    return all_bins

def  plot_all_binarize(gray_img: np.ndarray):
    images = all_binarize_algos(gray_img)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(algorithms[i], fontsize=10)
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig("all_bin.png")
    plt.show()
    

def binarize(gray_img: np.ndarray, func="WOLF", debug=False):
    binary_image = np.empty(gray_img.shape, gray_img.dtype)
    facsimile = doxapy.Binarization(getattr(doxapy.Binarization.Algorithms, func))
    facsimile.initialize(gray_img)
    facsimile.to_binary(binary_image)

    if debug:
        plt.imshow(binary_image, cmap="gray")
        skimage.io.imsave("./temp.png", binary_image)

    return binary_image

def otsu_with_mask(image, mask, thresh_adjust=0, debug=False):
    """
    Perform Otsu thresholding using only masked pixels.

    Parameters:
        image : ndarray (uint8)
            Single-channel image (e.g., saturation channel)
        mask : ndarray (uint8 or bool)
            Mask where non-zero / True pixels are used for Otsu
        thresh_adjust : int
            Manual threshold adjustment (clamped ±50)
        debug : bool
            Show debug plots

    Returns:
        binary : ndarray (uint8)
            Thresholded image (0 / 255)
        final_thresh : int
            Threshold used
    """
    if image.ndim != 2:
        raise ValueError("image must be single-channel")

    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Extract masked pixels
    masked_pixels = image[mask]

    if masked_pixels.size == 0:
        raise ValueError("Mask contains no valid pixels")

    # Compute histogram manually
    hist = cv2.calcHist([masked_pixels], [0], None, [256], [0, 256])
    hist = hist.ravel()

    # Manual Otsu computation
    total = masked_pixels.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    otsu_thresh = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) ** 2

        if var_between > max_var:
            max_var = var_between
            otsu_thresh = t

    # Clamp adjustment
    thresh_adjust = max(-50, min(50, thresh_adjust))
    final_thresh = int(np.clip(otsu_thresh + thresh_adjust, 0, 255))

    # Apply threshold to full image
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > final_thresh] = 255

    # Optional debug plots
    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        ax[1].plot(hist)
        ax[1].axvline(final_thresh, color='r', linestyle='--')
        ax[1].set_title("Masked Histogram")

        ax[2].imshow(binary, cmap='gray')
        ax[2].set_title(f"Thresholded (T={final_thresh})")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

    return binary, final_thresh


if __name__ == "__main__":
    img_file = "./image_stitching/data/heidelberg/1.png"
    # img_file = "./image_stitching/data/mma_krater/Screenshot 2025-12-11 at 5.25.52 PM.png"
    gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    plot_all_binarize(gray)
    # img = cv2.imread(img_file)
    # mask = segment_saturation(img)
    # kernel = np.ones((15, 15), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # contour_line, mask_clean = find_contour(mask)
    # binary, thresh = otsu_with_mask(
    #     gray,
    #     mask_clean,
    #     thresh_adjust=0,
    #     debug=True
    # )


