import os
import cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_bbox(size, box):
    W, H = size
    x1, y1, w, h = box
    xc = x1 + w / 2
    yc = y1 + h / 2
    return xc / W, yc / H, w / W, h / H

def process_split(split):
    images_dir = f"datasets/wider_face/WIDER_{split}/images"
    label_file = f"datasets/wider_face/wider_face_split/wider_face_{split}_bbx_gt.txt"

    out_img = f"datasets/wider_yolo/{split}/images"
    out_lbl = f"datasets/wider_yolo/{split}/labels"

    ensure_dir(out_img)
    ensure_dir(out_lbl)

    print(f"Processing {split}...")

    with open(label_file, "r") as f:
        lines = f.read().strip().split("\n")

    idx = 0
    total = len(lines)

    while idx < total:
        img_name = lines[idx].strip()
        idx += 1

        if idx >= total:
            break

        try:
            num_faces = int(lines[idx].strip())
        except:
            # Skip corrupted entries
            idx += 1
            continue

        idx += 1

        img_path = os.path.join(images_dir, img_name)
        out_img_path = os.path.join(out_img, img_name)

        if not os.path.exists(img_path):
            # Skip invalid paths
            for _ in range(num_faces):
                idx += 1
            continue

        ensure_dir(os.path.dirname(out_img_path))
        img = cv2.imread(img_path)

        if img is None:
            for _ in range(num_faces):
                idx += 1
            continue

        H, W = img.shape[:2]

        cv2.imwrite(out_img_path, img)

        label_path = os.path.join(out_lbl, img_name.replace(".jpg", ".txt"))
        ensure_dir(os.path.dirname(label_path))

        with open(label_path, "w") as out:
            for _ in range(num_faces):
                parts = lines[idx].strip().split()
                idx += 1

                if len(parts) < 4:
                    continue

                x1, y1, w, h = map(int, parts[:4])

                if w <= 0 or h <= 0:
                    continue

                xc, yc, w_norm, h_norm = convert_bbox((W, H), (x1, y1, w, h))
                out.write(f"0 {xc} {yc} {w_norm} {h_norm}\n")

    print(f"{split} done.")

process_split("train")
process_split("val")
