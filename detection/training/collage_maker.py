import os
import random
from PIL import Image

# --------------------------------------------------------
# Your 18 classes (same order as in your YOLO data.yaml)
# --------------------------------------------------------
CLASSES = [
    'bread', 'capsicum', 'carrot', 'chicken', 'coconuts', 'dal',
    'drumstick', 'eggs', 'mushroom', 'noodles',
    'onion', 'paneer', 'potatoes', 'soya', 'spinach', 'tomato'
]

# --------------------------------------------------------
# INPUT: your raw ingredient images
# DETECTOR/pictures/<class>/<images>
# --------------------------------------------------------
SOURCE_DIR = "pictures"

# --------------------------------------------------------
# OUTPUT: YOLO training dataset folders
# --------------------------------------------------------
OUT_IMG = "dataset/train/images"
OUT_LABEL = "dataset/train/labels"

COLLAGES_TO_GENERATE = 80
COLLAGE_SIZE = 640
GRID = 2   # 2×2 grid → 4 cells


def get_random_image(class_name):
    folder = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(folder):
        print(f"Missing folder for class: {class_name}")
        return None
    
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"No images in {folder}")
        return None
    
    return os.path.join(folder, random.choice(files))


def create_collage(img_paths, class_names):
    collage = Image.new("RGB", (COLLAGE_SIZE, COLLAGE_SIZE))
    tile = COLLAGE_SIZE // GRID

    # grid positions (x,y)
    positions = [(x * tile, y * tile) for y in range(GRID) for x in range(GRID)]

    yolo_labels = []

    for i, (img_path, cls_name) in enumerate(zip(img_paths, class_names)):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((tile, tile))

        x_offset, y_offset = positions[i]
        collage.paste(img, (x_offset, y_offset))

        # YOLO label → cx, cy, w, h (normalized)
        cx = (x_offset + tile / 2) / COLLAGE_SIZE
        cy = (y_offset + tile / 2) / COLLAGE_SIZE
        w = tile / COLLAGE_SIZE
        h = tile / COLLAGE_SIZE

        class_id = CLASSES.index(cls_name)
        yolo_labels.append(f"{class_id} {cx} {cy} {w} {h}")

    return collage, yolo_labels


# --------------------------------------------------------
# MAIN LOOP: Generate Collages
# --------------------------------------------------------
for i in range(COLLAGES_TO_GENERATE):
    num_objects = random.randint(2, 4)  # choose 2–4 ingredients
    selected_classes = random.sample(CLASSES, num_objects)

    img_paths = []
    valid_classes = []

    for cls in selected_classes:
        p = get_random_image(cls)
        if p:
            img_paths.append(p)
            valid_classes.append(cls)

    if not img_paths:
        continue

    collage, labels = create_collage(img_paths, valid_classes)

    img_name = f"collage_{i}.jpg"
    label_name = f"collage_{i}.txt"

    collage.save(os.path.join(OUT_IMG, img_name))

    with open(os.path.join(OUT_LABEL, label_name), "w") as f:
        f.write("\n".join(labels))

print(f"\n✔ Successfully generated {COLLAGES_TO_GENERATE} collage images!")
