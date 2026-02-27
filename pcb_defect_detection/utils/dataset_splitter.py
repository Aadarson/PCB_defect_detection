import os
import shutil
import random
import yaml
from math import floor

from utils.logger import system_logger

def split_dataset(
    source_images_dir: str, 
    source_labels_dir: str, 
    output_dir: str, 
    classes: list,
    split_ratio: tuple=(0.8, 0.2), 
    seed: int=42
):
    """
    Splits a flat dataset into YOLOv8 expected structure (train/val).
    output_dir/
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    random.seed(seed)
    system_logger.info(f"Splitting dataset. Seed = {seed}, Ratio = {split_ratio}")

    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_out, split), exist_ok=True)
        os.makedirs(os.path.join(labels_out, split), exist_ok=True)

    # Gather images
    all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_images)

    train_count = floor(len(all_images) * split_ratio[0])
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:]

    system_logger.info(f"Total: {len(all_images)} | Train: {len(train_images)} | Val: {len(val_images)}")

    def move_files(file_list, target_split):
        for img_name in file_list:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"

            src_img = os.path.join(source_images_dir, img_name)
            src_lbl = os.path.join(source_labels_dir, lbl_name)
            
            dst_img = os.path.join(images_out, target_split, img_name)
            dst_lbl = os.path.join(labels_out, target_split, lbl_name)

            if os.path.exists(src_lbl):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
            else:
                system_logger.warning(f"Label missing for {img_name}, skipping.")

    move_files(train_images, "train")
    move_files(val_images, "val")

    # Generate data.yaml
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)}
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
        
    system_logger.info(f"Dataset split complete. `data.yaml` generated at {data_yaml_path}")

if __name__ == "__main__":
    # Example usage for manual trigger
    print("Run this passing source paths inside a custom script.")
