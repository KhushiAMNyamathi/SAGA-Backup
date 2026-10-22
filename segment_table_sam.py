import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

IMAGE_PATH = "/home/adminusr/Desktop/khushi/images/21.png"
CHECKPOINT = os.path.join(
    os.getcwd(),
    "third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"
)
OUTPUT_DIR = "/home/adminusr/Desktop/khushi/table_segmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading image...")
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("Loading SAM model on GPU...")
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
sam.to(device="cuda")
predictor = SamPredictor(sam)

print("Preparing image for SAM...")
predictor.set_image(image)

print("Image size:", image.shape[1], "x", image.shape[0])
print("Click once near the CENTER of the white table.")
print("Close the image window when you are satisfied with the mask.")

points = []
labels = []
current_mask = None

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(image)
ax.set_title("Left-click: table | Right-click: background | Close when done")

overlay = None

def redraw():
    global overlay
    ax.clear()
    ax.imshow(image)

    if current_mask is not None:
        colored = np.zeros((*current_mask.shape, 4), dtype=np.float32)
        colored[current_mask] = [0.0, 1.0, 0.0, 0.40]
        ax.imshow(colored)

    for (x, y), label in zip(points, labels):
        color = "lime" if label == 1 else "red"
        ax.scatter(x, y, c=color, s=80, marker="+")

    ax.set_title("Left-click: table | Right-click: background")
    ax.axis("off")
    fig.canvas.draw_idle()

def on_click(event):
    global current_mask

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)

    if event.button == 1:
        label = 1
    elif event.button == 3:
        label = 0
    else:
        return

    points.append([x, y])
    labels.append(label)

    coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=point_labels,
        multimask_output=True
    )

    best = int(np.argmax(scores))
    current_mask = masks[best]

    print(
        f"Added {'table' if label == 1 else 'background'} point "
        f"({x}, {y}); SAM score={scores[best]:.3f}"
    )

    redraw()

    mask_path = os.path.join(OUTPUT_DIR, "table_mask.png")
    preview_path = os.path.join(OUTPUT_DIR, "table_preview.png")

    cv2.imwrite(mask_path, current_mask.astype(np.uint8) * 255)

    preview = image.copy()
    tint = np.zeros_like(preview)
    tint[:, :, 1] = 255
    preview[current_mask] = (
        0.6 * preview[current_mask] + 0.4 * tint[current_mask]
    ).astype(np.uint8)

    cv2.imwrite(
        preview_path,
        cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    )

    print("Saved mask:", mask_path)
    print("Saved preview:", preview_path)

fig.canvas.mpl_connect("button_press_event", on_click)
plt.tight_layout()
plt.show()

print("Finished. Output folder:", OUTPUT_DIR)
