from pathlib import Path
import os

home_dir = os.environ["HOME"]
pictures = Path(home_dir) / "Pictures" 
png_files = pictures.glob("**/*.png")
print(type(png_files))

for f in png_files:
    print(f)

