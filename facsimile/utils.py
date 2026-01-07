import os
import uuid
import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_image(path, as_gray=False):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def save_image_cv(img, out_path):
    base, ext = os.path.splitext(out_path)
    ext = ".png"
    out_path = base + ext
    success, imbuf = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Failed to encode image as PNG {out_path}")
    imbuf.tofile(out_path)