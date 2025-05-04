import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel

logging.set_verbosity_error()

# 数据路径（Kaggle COCO 2017）
train_ann_path = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
val_ann_path = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'
train_img_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017/'
val_img_dir = '/kaggle/input/coco-2017-dataset/coco2017/val2017/'
features_dir = '/kaggle/working/features/'  # 输出目录更新为 /kaggle/working/

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name)
clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)

def load_data():
    data = {'train': [], 'val': []}
    
    def process_annotation(json_path, split_name):
        with open(json_path, 'r') as f:
            ann = json.load(f)
        images_info = ann['images']
        for img in images_info:
            data[split_name].append({
                'file_name': img['file_name'],
                'cocoid': img['id']
            })

    process_annotation(train_ann_path, 'train')
    process_annotation(val_ann_path, 'val')
    return data

def encode_split(data, split, img_dir):
    df = pd.DataFrame(data[split])

    bs = 256
    os.makedirs(features_dir, exist_ok=True)
    h5_file_path = os.path.join(features_dir, f'{split}.hdf5')
    if os.path.exists(h5_file_path):
        print(f"Skipping {split}, feature file already exists at: {h5_file_path}")
        return
    h5py_file = h5py.File(h5_file_path, 'w')

    for idx in tqdm(range(0, len(df), bs), desc=f'Encoding {split}'):
        batch = df.iloc[idx:idx + bs]
        images = []
        valid_ids = []

        for file_name, cocoid in zip(batch['file_name'], batch['cocoid']):
            img_path = os.path.join(img_dir, file_name)
            try:
                image = Image.open(img_path).convert("RGB")
                images.append(image)
                valid_ids.append(cocoid)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

        if not images:
            continue

        with torch.no_grad():
            pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
            encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy()

        for cocoid, encoding in zip(valid_ids, encodings):
            h5py_file.create_dataset(str(cocoid), (50, 768), data=encoding)

    h5py_file.close()

# 执行
data = load_data()
encode_split(data, 'train', train_img_dir)
encode_split(data, 'val', val_img_dir)
