# ppmtoimg.py
# Converts the ray traced render.ppm image to render.png using Pillow (PIL)

from PIL import Image
import os

INPUT_FILE = "render.ppm"
OUTPUT_FILE = "render.png"

def convert_ppm_to_png(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' does not exist!")
        return

    try:
        img = Image.open(input_path)
        img.save(output_path)
        print(f"✅ Conversion complete! Saved as '{output_path}'")
    except Exception as e:
        print(f"❌ Failed to convert image: {e}")

if __name__ == "__main__":
    convert_ppm_to_png(INPUT_FILE, OUTPUT_FILE)
