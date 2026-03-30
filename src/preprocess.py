import os
import json

def remove_duplicate_images(json_file):

    with open(json_file, "r") as f:
        images_to_remove = json.load(f)

    removed = 0

    for img_path in images_to_remove:

        if os.path.exists(img_path):
            os.remove(img_path)
            removed += 1
        else:
            print(f"File not found: {img_path}")

    print(f"Removed {removed} duplicate images")


if __name__ == "__main__":
    remove_duplicate_images("images_to_remove.json")