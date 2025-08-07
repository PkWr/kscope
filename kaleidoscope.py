import os
import cv2
import numpy as np
import random
import time
from PIL import Image
import imageio.v2 as imageio

# --- Config ---
image_dir = "images"
output_dir = "output"
frame_count = 240
frame_size = 512
max_iterations = 3  # Set your desired number of loops

# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

iteration = 0
while iteration < max_iterations:
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()

    # --- Set random segment count between 2 and len(image_files) ---
    n_segments = random.randint(2, len(image_files))
    rotation_per_frame = 360 / frame_count

    if len(image_files) < n_segments:
        raise ValueError(f"Need at least {n_segments} images in the images/ folder. Found {len(image_files)}.")

    # Preprocess all images: crop, resize, convert to greyscale
    processed_images = []
    for filename in image_files:
        path = os.path.join(image_dir, filename)
        img = Image.open(path).convert("L")
        min_side = min(img.size)
        img = img.crop(((img.width - min_side) // 2, (img.height - min_side) // 2,
                        (img.width + min_side) // 2, (img.height + min_side) // 2))
        img = img.resize((frame_size, frame_size))
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        processed_images.append(img_np)

    h, w = frame_size, frame_size
    center = (w // 2, h // 2)
    angle_per_slice = 360 / n_segments
    frames = []

    # --- Circular Frame Mask ---
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, w // 2, 255, -1)
    circle_mask_rgb = cv2.merge([circle_mask] * 3)

    # --- Generate Animated Frames ---
    for frame in range(frame_count):
        output = np.zeros((h, w, 3), dtype=np.uint8)
        base_rotation = frame * rotation_per_frame

        for i in range(n_segments):
            src_img = processed_images[i % len(processed_images)]

            mask = np.zeros((h, w), dtype=np.uint8)
            angle1 = base_rotation + i * angle_per_slice
            angle2 = base_rotation + (i + 1) * angle_per_slice

            pts = np.array([
                center,
                (
                    int(center[0] + w * np.cos(np.radians(angle1))),
                    int(center[1] + h * np.sin(np.radians(angle1)))
                ),
                (
                    int(center[0] + w * np.cos(np.radians(angle2))),
                    int(center[1] + h * np.sin(np.radians(angle2)))
                )
            ])
            cv2.fillConvexPoly(mask, pts, 255)

            wedge = cv2.bitwise_and(src_img, src_img, mask=mask)
            M = cv2.getRotationMatrix2D(center, -angle1, 1.0)
            rotated = cv2.warpAffine(wedge, M, (w, h))

            if i % 2 == 1:
                rotated = cv2.flip(rotated, 1)

            output = cv2.add(output, rotated)

        # Apply circular mask
        output = cv2.bitwise_and(output, circle_mask_rgb)
        frames.append(Image.fromarray(output))

    # --- Save GIF with incrementing filename ---
    output_gif = os.path.join(output_dir, f"kaleidoscope_{iteration:03}.gif")
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"✅ Saved kaleidoscope wedge-blended animation with {n_segments} segments to {output_gif}")

    iteration += 1
    time.sleep(3)

# --- Write auto-cycling HTML page with fullscreen toggle ---
html_path = os.path.join(output_dir, "index.html")
gif_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".gif")])

with open(html_path, "w") as f:
    f.write("""
    <html>
    <head>
        <title>Kaleidoscope Viewer</title>
        <meta charset='UTF-8'>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                background: #111;
                color: #eee;
                font-family: sans-serif;
                overflow: hidden;
            }
            #fullscreen-toggle {
                position: absolute;
                top: 10px;
                right: 10px;
                padding: 10px 15px;
                background: #222;
                color: #fff;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                z-index: 10;
            }
        </style>
        <script>
            let gifs = [
    """)
    for gif in gif_files:
        f.write(f"                '{gif}',\n")
    f.write("""            ];
            let index = 0;
            function cycle() {
                document.getElementById('kaleido').src = gifs[index];
                index = (index + 1) % gifs.length;
            }
            setInterval(cycle, 5000);

            function toggleFullscreen() {
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen();
                } else {
                    document.exitFullscreen();
                }
            }
        </script>
    </head>
    <body onclick="toggleFullscreen()">
        <button id='fullscreen-toggle'>Toggle Fullscreen</button>
        <img id='kaleido' src='""" + gif_files[0] + """' style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:90vmin;border-radius:50%;'>
    </body>
    </html>
    """)
