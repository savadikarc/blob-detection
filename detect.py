import argparse
import numpy as np
import cv2
from src.log_blobs import detect_bolbs, draw_circles

ap = argparse.ArgumentParser()
ap.add_argument("-image", "--input-image", required=True, help="Absolute path to the input image")
ap.add_argument("-output", "--save-filename", required=True, help="Absolute path to the file for saving the circels image")
ap.add_argument("--sigma", default=1.6, type=float, help="Base sigma value")
ap.add_argument("--scale-factor", default=float(np.sqrt(2)), type=float, help="Scale multiplier")
ap.add_argument("--num-scales", default=9, type=int, help="Number of scales to generate")
ap.add_argument("--threshold", default=0.01, type=float, help="Threshold for the detected maxima to be considered valid")
args = ap.parse_args()

image_file = args.input_image

scale = args.sigma
scale_factor = args.scale_factor
num_scales = args.num_scales
threshold = args.threshold

print(f"---\nProcessing {image_file}")

image = cv2.imread(image_file)
if image is None:
    raise FileNotFoundError(f"File {image_file} not found.")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blob_centers = detect_bolbs(image_gray, scale, num_scales, scale_factor, threshold, use_dog=False)

# plot circles, skip the first and last scales
plot_image = draw_circles(image_gray, blob_centers, scale, scale_factor, show=False, save=True, filename=args.save_filename)
print("---")