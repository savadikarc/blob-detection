import time
import os

SEP = os.path.sep

# Get the absolute path of the file
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir, "")

input_base_dir = os.path.join(file_dir, "..", "")
output_base = os.path.join(file_dir, "..", "output_8_images", "")
if not os.path.exists(output_base):
    os.makedirs(output_base)

os.system(f"pip install -r {file_dir}requirements.txt")

start_time = time.time()
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}butterfly.jpg --save-filename {output_base}butterfly.png --threshold 0.01")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}sunflowers.jpg --save-filename {output_base}sunflowers.png --threshold 0.01")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}fishes.jpg --save-filename {output_base}fishes.png --threshold 0.009")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}einstein.jpg --save-filename {output_base}einstein.png --threshold 0.02")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}spotted_catbird.png --save-filename {output_base}spotted_catbird.png --threshold 0.015")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}flower.png --save-filename {output_base}flower.png --threshold 0.015")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}wolf.png --save-filename {output_base}wolf.png --threshold 0.02")
os.system(f"python {file_dir}detect.py --input-image {input_base_dir}images{SEP}dog.png --save-filename {output_base}dog.png --threshold 0.02")

print(f"-----\nProcessed 8 images in {time.time() - start_time} seconds")
