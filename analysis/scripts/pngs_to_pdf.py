from PIL import Image
from pathlib import Path

# Collect images: layout desired: large page image first, then the four analysis PNGs
images = []
root = Path('.')
page_img = root / 'report_images_v2' / 'page_01.png'
plots = sorted((root / 'report_plots_final').glob('*.png'))

if not page_img.exists():
    print('Main page image not found:', page_img)
    raise SystemExit(1)

images.append(Image.open(page_img).convert('RGB'))
for p in plots:
    images.append(Image.open(p).convert('RGB'))

out = root / 'report_final_combined.pdf'
images[0].save(out, save_all=True, append_images=images[1:], quality=95)
print('Saved combined PDF:', out)
