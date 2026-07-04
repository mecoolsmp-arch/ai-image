import os
import tempfile
from gradio_client import Client, handle_file
from PIL import Image, ImageDraw

img = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(img)
d.rectangle([100, 100, 400, 400], fill="black")
d.text([130, 240], "manga", fill="red")
tmp = os.path.join(tempfile.gettempdir(), "mt.png")
img.save(tmp)

c = Client("http://127.0.0.1:7860", verbose=False)
try:
    r = c.predict(
        handle_file(tmp),
        "turn this manga into a realistic photo",
        "anime, manga",
        "Z-Image Turbo (Int8 - 8GB Safe)",
        768,
        0.72,
        4,
        -1,
        "cuda",
        True,
        api_name="/manga_to_realistic",
    )
    print("RESULT:", str(r)[:400])
except Exception as e:
    import traceback
    traceback.print_exc()
    print("CALL FAILED:", e)
