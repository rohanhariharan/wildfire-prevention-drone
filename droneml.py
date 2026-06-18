import os
import yaml
from datasets import load_dataset
from tqdm import tqdm
from ultralytics import YOLO

def prepare_yolo_data():
    dataset_name = "pyronear/pyro-sdis"
    base_dir = "datasets/pyro_sdis"
    
    print(f"extracting images and labels from pyro-sdis...")
    dataset = load_dataset(dataset_name)
    
    for split_name in ['train', 'val']:
        img_dir = os.path.join(base_dir, "images", split_name)
        lbl_dir = os.path.join(base_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"\nWriting labels for the '{split_name}' split...")
        for idx, row in enumerate(tqdm(dataset[split_name])):
            img = row['image']
            img_name = row.get('image_name', f"img_{idx}.jpg")
            base_name = os.path.splitext(img_name)[0]
            
            #save image
            img_path = os.path.join(img_dir, f"{base_name}.jpg")
            img.convert("RGB").save(img_path)
            
            #extract annotations in yolo format
            annotation_str = row.get('annotations', "")
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            with open(lbl_path, "w") as f:
                if annotation_str and isinstance(annotation_str, str):
                    f.write(annotation_str.strip() + "\n")
                    
    # Step 2: Create YAML configuration matching the dataset indices
    print("\ncreating dataset.yaml configuration file...")
    yaml_content = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "smoke"       
        }
    }
    
    yaml_path = "pyro_sdis.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_content, f, default_flow_style=False)
        
    return yaml_path

def train_yolo(yaml_config_path):
    print("\nstarting yolo26n training...")
    model = YOLO("[YOLO PATH HERE]")
    
    model.train(
        data=yaml_config_path,
        epochs=13,
        imgsz=640,
        batch=36,
        device=0, 
        workers=4,
        resume=True
    )

if __name__ == "__main__":
    config_file = prepare_yolo_data()
    train_yolo(config_file)
