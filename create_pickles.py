#!/usr/bin/env python3
"""
Create pickle files for APTOS2019 and HAM1000 datasets
Saves pickle files in the same directories as the raw data
"""

import pandas as pd
import pickle
import os
import numpy as np
from pathlib import Path
from collections import Counter
import glob

def create_aptos_pickles():
    """
    Create APTOS2019 pickle files from CSV + images
    Save in the same APTOS2019 directory
    """
    print("Processing APTOS2019 dataset...")

    base_path = "dataset/APTOS2019"
    split_csv_map = {
        "train": f"{base_path}/train.csv",
        "val": f"{base_path}/val.csv",
        "test": f"{base_path}/test.csv",
    }
    split_image_dir_map = {
        "train": f"{base_path}/train_images",
        "val": f"{base_path}/val_images",
        "test": f"{base_path}/test_images",
    }

    split_pickles = {
        "train": f"{base_path}/aptos_train.pkl",
        "val": f"{base_path}/aptos_val.pkl",
        "test": f"{base_path}/aptos_test.pkl",
    }

    summary = {}

    for split_name, csv_path in split_csv_map.items():
        if not os.path.exists(csv_path):
            print(f"Missing CSV for {split_name}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"APTOS {split_name.title()} samples: {len(df)}")

        image_subdir = "train_images"
        if split_name == "test":
            image_subdir = "test_images"
        elif split_name == "val" and os.path.exists(f"{base_path}/val_images"):
            image_subdir = "val_images"
        image_dir = split_image_dir_map.get(split_name, f"{base_path}/{image_subdir}")

        data = []
        missing = 0
        label_counter = Counter()

        for _, row in df.iterrows():
            image_id = row['id_code']
            label = int(row.get('diagnosis', 0))

            img_path = f"{image_subdir}/{image_id}.png"
            abs_img_path = f"{image_dir}/{image_id}.png"

            if os.path.exists(abs_img_path):
                data.append({
                    'img_root': img_path,
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

def create_ham1000_pickles():
    """
    Create HAM1000 pickle files from CSV + images
    Save in the same HAM1000 directory
    """
    print("\nProcessing HAM1000 dataset...")
    
    # Paths - save in same directory as raw data
    base_path = "dataset/HAM1000"
    train_csv = f"{base_path}/Training_GroundTruth/Training_GroundTruth.csv"
    test_csv = f"{base_path}/Test_GroundTruth/Test_GroundTruth.csv"
    train_images_dir = f"{base_path}/Training_Input"
    test_images_dir = f"{base_path}/Test_Input"
    
    # Class names mapping
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    # Read training data
    train_df = pd.read_csv(train_csv)
    print(f"HAM1000 Training samples: {len(train_df)}")
    
    # Create training pickle data
    train_data = []
    missing_train = 0
    
    for idx, row in train_df.iterrows():
        image_id = row['image']
        
        # Find the class with value 1.0 (one-hot encoded)
        label = -1
        for i, class_name in enumerate(class_names):
            if row[class_name] == 1.0:
                label = i
                break
        
        if label == -1:
            print(f"No valid label found for {image_id}")
            continue
        
        # Image path - relative to where pickle will be saved
        img_path = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            abs_potential_path = f"{train_images_dir}/{image_id}{ext}"
            if os.path.exists(abs_potential_path):
                img_path = f"Training_Input/{image_id}{ext}"  # Relative path
                break
        
        if img_path:
            train_data.append({
                'img_root': img_path,  # Relative path
                'label': label
            })
        else:
            missing_train += 1
            print(f"Missing training image: {image_id}")
    
    print(f"HAM1000 Training: {len(train_data)} samples, {missing_train} missing")
    
    # Read test data
    test_df = pd.read_csv(test_csv)
    print(f"HAM1000 Test samples: {len(test_df)}")
    
    # Create test pickle data
    test_data = []
    missing_test = 0
    
    for idx, row in test_df.iterrows():
        image_id = row['image']
        
        # Find the class with value 1.0 (one-hot encoded)
        label = -1
        for i, class_name in enumerate(class_names):
            if row[class_name] == 1.0:
                label = i
                break
        
        if label == -1:
            print(f"No valid label found for {image_id}")
            continue
        
        # Image path - relative to where pickle will be saved
        img_path = None
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            abs_potential_path = f"{test_images_dir}/{image_id}{ext}"
            if os.path.exists(abs_potential_path):
                img_path = f"Test_Input/{image_id}{ext}"  # Relative path
                break
        
        if img_path:
            test_data.append({
                'img_root': img_path,  # Relative path
                'label': label
            })
        else:
            missing_test += 1
            print(f"Missing test image: {image_id}")
    
    print(f"HAM1000 Test: {len(test_data)} samples, {missing_test} missing")
    
    # Save pickle files IN THE SAME DIRECTORY
    with open(f"{base_path}/isic2018_train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    print(f"Saved: {base_path}/isic2018_train.pkl")
    
    with open(f"{base_path}/isic2018_test.pkl", "wb") as f:
        pickle.dump(test_data, f)
    print(f"Saved: {base_path}/isic2018_test.pkl")
    
    # Print class distribution
    train_labels = [item['label'] for item in train_data]
    for class_id, class_name in enumerate(class_names):
        count = train_labels.count(class_id)
        print(f"   Class {class_id} ({class_name}): {count} samples ({count/len(train_labels)*100:.1f}%)")
    
    return True

def main():
    print("Creating co-located dataset pickle files for DiffMICv2")
    print("=" * 60)
    
    success = True
    
    # Create APTOS pickles
    try:
        success &= create_aptos_pickles()
    except Exception as e:
        print(f"Error creating APTOS pickles: {e}")
        success = False
    
    # Create HAM1000 pickles  
    try:
        success &= create_ham1000_pickles()
    except Exception as e:
        print(f"Error creating HAM1000 pickles: {e}")
        success = False
    
    if success:
        print("\nAll dataset pickle files created successfully!")
        print("\nNext steps:")
        print("1. Update config files to point to new locations")
        print("2. Run: python get_specific_pretrained_models.py")
        print("3. Train APTOS: python train_aptos.py")
        print("4. Train ISIC: python train_isic.py")
    else:
        print("\nSome errors occurred. Please check the output above.")
    
    return success

if __name__ == "__main__":
    main()