import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode

def find_contour(mask, debug=False):
    # find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"  Found {len(contours)} contour(s)")
    assert len(contours) > 0, "No contours found — boundary detection failed."

    # get largest contour and fill it in
    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    contour_line = np.zeros_like(mask)
    cv2.drawContours(contour_line, [largest_contour], -1, 255, thickness=2)
    
    if debug:
        # draw contour on img
        if len(mask.shape) == 2:
            outline = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            outline = mask.copy()
        cv2.drawContours(outline, [largest_contour], -1, (0, 0, 255), 2)

        cv2.imshow("Original", mask)
        cv2.imshow("Boundary mask (raw)", mask)
        cv2.imshow("Largest Contour Filled Mask", filled_mask)
        cv2.imshow("Largest Contour Outlined", outline)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contour_line, filled_mask

def kmeans_segment(img, features, k=2, r=10):
    h, w = img.shape[:2]

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features)
    labels = kmeans.labels_.reshape(h, w)

    # Find majority label in rxr center
    cx, cy = w // 2, h // 2
    roi = labels[cy - r:cy + r, cx - r:cx + r]

    m = mode(roi.flatten(), keepdims=False).mode
    majority_label = int(m)

    binary = np.uint8(labels == majority_label) * 255
    return labels, binary, majority_label


def segment_hsv(img, select_hsv="HSV", output_file=None, debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Pre-filtering ---
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_blur = cv2.medianBlur(img_blur, 5)

    h, w = img_blur.shape[:2]
    rgb = img_blur.reshape(-1, 3)

    # HSV feature set
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    hsv = hsv_img.reshape(-1, 3)
    
    if select_hsv == "H":
        hsv_selected = hsv[:, [0]]
    elif select_hsv == "S":
        hsv_selected = hsv[:, [1]]
    elif select_hsv == "V":
        hsv_selected = hsv[:, [2]]
    else:
        hsv_selected = hsv

    # RGB + HSV kmeans
    rgb_hsv = np.column_stack((rgb, hsv_selected))
    labels, mask, maj_labels = kmeans_segment(
        img_blur, rgb_hsv, k=2
    )
    # close holes
    # kernel = np.ones((morph_close, morph_close), np.uint8)
    # mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imshow("Mask via Kmeans with seHSVlect", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # plot
        # fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        # axes[0].set_title("Original")
        # axes[0].imshow(img_blur)
        # axes[0].axis("off")

        # axes[1].set_title("KMeans (RGB + %s)" % select_hsv)
        # axes[1].imshow(labels, cmap="nipy_spectral")
        # axes[1].axis("off")


        # axes[2].set_title("KMeans (RGB + %s) binary" % select_hsv)
        # axes[2].imshow(mask, cmap="nipy_spectral")
        # axes[2].axis("off")

        # plt.tight_layout()

        # save if file name
        # if output_file:
        #     plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.show()
        # plt.close()
    return mask

def segment_saturation(img, thresh_adjust:int=0, debug=False):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv_image[:, :, 1]
    otsu_thresh, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    manual_thresh = np.clip(otsu_thresh + thresh_adjust, 0, 255)
    _, binary = cv2.threshold(saturation, manual_thresh, 255, cv2.THRESH_BINARY)
    
    if debug:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
        ax = axes.ravel()
        
        ax[0].imshow(saturation, cmap='gray')
        ax[0].set_title('Saturation Channel')
        ax[0].axis('off')
        
        ax[1].hist(saturation.ravel(), bins=256)
        ax[1].axvline(_, color='r', linestyle='--')  # Threshold value
        ax[1].set_title('Histogram')
        
        ax[2].imshow(binary, cmap='gray')
        ax[2].set_title('Thresholded')
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    return binary

def segment_otsu(gray_img, thresh_adjust:int=0, debug=False):
    gray_image = img[:, :, 1]
    otsu_thresh, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    manual_thresh = int(np.clip(otsu_thresh + thresh_adjust, 0, 255))
     # Decide inversion automatically (light background)
    invert = True#gray_img.mean() > manual_thresh
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    _, binary = cv2.threshold(gray_image, manual_thresh, 255, thresh_type)
    
    if debug:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
        ax = axes.ravel()
        
        ax[0].imshow(gray_image, cmap='gray')
        ax[0].set_title('Saturation Channel')
        ax[0].axis('off')
        
        ax[1].hist(gray_image.ravel(), bins=256)
        ax[1].axvline(_, color='r', linestyle='--')  # Threshold value
        ax[1].set_title('Histogram')
        
        ax[2].imshow(binary, cmap='gray')
        ax[2].set_title('Thresholded')
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    return binary

def process_single(img, method="color", thresh_adjust=0, debug=True):
    if method=="saturation":
        segmented = segment_saturation(img, thresh_adjust, debug=debug)
    elif method=="kmeans":
        segmented = segment_hsv(img, debug=debug)
    elif method=="otsu":
        segmented = segment_otsu(img, thresh_adjust=60, debug=debug)
    else:
        print("invalid method")
        return
    if debug:
        cv2.imshow("new mask", segmented)
        # cv2.imwrite("mask.png",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contour_line, mask = find_contour(segmented)

    if debug:
        cv2.imshow("new mask", mask)
        cv2.imwrite("mask.png",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Extract object
    result = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow("Segmented object", result)
        # cv2.imwrite("segmented.png",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # img_file = "./image_stitching/data/mma_krater/Screenshot 2025-12-11 at 5.25.52 PM.png"
    # img_file = "./image_stitching/data/heidelberg/1.png"
    img_file = "./facsimile/data/test30_imgs/original-1-7.JPG"
    img = cv2.imread(img_file)
    # print(img)
    process_single(img, method="saturation", thresh_adjust=-10, debug=True)