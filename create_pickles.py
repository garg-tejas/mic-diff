import pandas as pd
import pickle
import os
import yaml
from pathlib import Path
from collections import Counter

def load_config(config_path='configs/aptos.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_aptos_pickles(config_path='configs/aptos.yml'):
    print("Processing APTOS2019 dataset...")
    
    config = load_config(config_path)
    data_config = config['data']
    
    split_csv_map = {
        "train": data_config['train_csv'],
        "val": data_config['val_csv'],
        "test": data_config['test_csv'],
    }
    split_image_dir_map = {
        "train": data_config['train_images'],
        "val": data_config['val_images'],
        "test": data_config['test_images'],
    }
    split_pickles = {
        "train": data_config['traindata'],
        "val": data_config['valdata'],
        "test": data_config['testdata'],
    }
    
    os.makedirs(os.path.dirname(data_config['traindata']), exist_ok=True)

    summary = {}

    for split_name, csv_path in split_csv_map.items():
        if not os.path.exists(csv_path):
            print(f"Missing CSV for {split_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"APTOS {split_name.title()} samples: {len(df)}")

        image_dir = split_image_dir_map[split_name]
        dataroot = data_config['dataroot']
        
        rel_image_dir = os.path.relpath(image_dir, dataroot) if dataroot else image_dir

        data = []
        missing = 0
        label_counter = Counter()

        for _, row in df.iterrows():
            image_id = row['id_code']
            label = int(row.get('diagnosis', 0))

            abs_img_path = os.path.join(image_dir, f"{image_id}.png")
            rel_img_path = os.path.join(rel_image_dir, f"{image_id}.png")

            if os.path.exists(abs_img_path):
                data.append({
                    'img_root': rel_img_path,
                    'label': label
                })
                label_counter[label] += 1
            else:
                missing += 1
                print(f"Missing {split_name} image: {abs_img_path}")

        if data:
            with open(split_pickles[split_name], "wb") as f:
                pickle.dump(data, f)
            print(f"Saved: {split_pickles[split_name]}")

            summary[split_name] = {
                'count': len(data),
                'missing': missing,
                'label_dist': dict(label_counter)
            }
        else:
            print(f"No data collected for {split_name}, skipping pickle save.")

    train_summary = summary.get('train')
    if train_summary and train_summary['label_dist']:
        total_train = train_summary['count']
        print("Training class distribution:")
        for class_id in range(5):
            count = train_summary['label_dist'].get(class_id, 0)
            pct = (count / total_train * 100) if total_train else 0
            print(f"   Class {class_id}: {count} samples ({pct:.1f}%)")

    return True

def main(config_path='configs/aptos.yml'):
    print("Creating dataset pickle files for DiffMICv2")
    print("=" * 60)
    
    success = True
    
    try:
        success &= create_aptos_pickles(config_path)
    except Exception as e:
        print(f"Error creating APTOS pickles: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\nAll dataset pickle files created successfully!")
    else:
        print("\nSome errors occurred. Please check the output above.")
    
    return success

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/aptos.yml'
    main(config_path)