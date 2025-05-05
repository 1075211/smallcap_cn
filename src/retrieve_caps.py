import json
from tqdm import tqdm
import jieba  # 中文分词
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_flickr8k_data(feature_dir, caption_path):
    """加载Flickr8k-CN预提取特征和描述"""
    # 1. 加载图像特征
    shape_path = os.path.join(feature_dir, "shape.txt")
    with open(shape_path, 'r') as f:
        shape = tuple(map(int, f.read().strip().split()))  # 改为空格分隔
    
    features = np.fromfile(
        os.path.join(feature_dir, "feature.bin"), 
        dtype=np.float32
    ).reshape(shape)
    
    # 2. 加载图像ID
    id_path = os.path.join(feature_dir, "id.txt")
    with open(id_path, 'r') as f:
        image_ids = [line.strip() for line in f]
    
    # 3. 加载中文描述
    captions = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)  # 只分割第一个空格
            if len(parts) == 2:
                img_id = parts[0].split('#')[0]
                caption = parts[1]
                captions.append({'image_id': img_id, 'caption': caption})
    
    return image_ids, features, captions

def filter_captions(captions, max_length=25):
    """过滤过长的中文描述（按分词后长度）"""
    filtered_image_ids, filtered_captions = [], []
    for item in captions:
        words = list(jieba.cut(item['caption']))  # 中文分词
        if len(words) <= max_length:
            filtered_image_ids.append(item['image_id'])
            filtered_captions.append(item['caption'])
    return filtered_image_ids, filtered_captions

def encode_captions(captions, model, device):
    """用CLIP编码中文描述（需预加载多语言CLIP模型）"""
    bs = 256
    encoded_captions = []
    for idx in tqdm(range(0, len(captions), bs), desc="Encoding captions"):
        batch = captions[idx:idx+bs]
        with torch.no_grad():
            # 注意：需使用支持中文的CLIP模型（如 'multilingual' 版本）
            input_ids = clip.tokenize(batch, truncate=True).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())
    return np.concatenate(encoded_captions)

def main():
    # 数据集路径（Flickr8k-CN）
    feature_dir = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/FeatureData/pyresnet152-pool5osl2"
    caption_path = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/TextData/seg.flickr8kzhbJanbosontrain.caption.txt"
    
    # 输出路径
    os.makedirs("/kaggle/working/datastore", exist_ok=True)
    captions_path = "/kaggle/working/datastore/filtered_captions.json"
    encoded_captions_path = "/kaggle/working/datastore/encoded_captions.npy"
    encoded_features_path = "/kaggle/working/datastore/encoded_features.npy"
    faiss_index_path = "/kaggle/working/datastore/flickr8k_index"
    retrieved_caps_path = "/kaggle/working/retrieved_caps_resnet152.json"

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Step 1: 加载数据
    print('Loading Flickr8k-CN data')
    image_ids, features, captions = load_flickr8k_data(feature_dir, caption_path)

    # Step 2: 过滤长描述
    if os.path.exists(captions_path):
        print('Loading filtered captions')
        with open(captions_path, 'r') as f:
            xb_image_ids, filtered_captions = json.load(f)
    else:
        print('Filtering long captions')
        xb_image_ids, filtered_captions = filter_captions(captions)
        with open(captions_path, 'w') as f:
            json.dump([xb_image_ids, filtered_captions], f)

    # Step 3: 编码描述（需支持中文的CLIP模型）
    clip_model, _ = clip.load("ViT-B/32", device=device)  # 替换为多语言模型（如 'M-CLIP'）
    if os.path.exists(encoded_captions_path):
        print('Loading encoded captions')
        encoded_captions = np.load(encoded_captions_path)
    else:
        print('Encoding captions')
        encoded_captions = encode_captions(filtered_captions, clip_model, device)
        np.save(encoded_captions_path, encoded_captions)

    # Step 4: 直接使用预提取图像特征
    print('Using pre-computed image features')
    encoded_features = features  # 直接使用ResNet152特征
    np.save(encoded_features_path, encoded_features)

    # Step 5: 构建FAISS索引并检索
    if os.path.exists(retrieved_caps_path):
        print('FAISS index and retrieval already done. Skipping.')
    else:
        print('Building FAISS index and retrieving neighbors')
        index, nns = get_nns(encoded_captions, encoded_features)
        retrieved_caps = filter_nns(nns, xb_image_ids, filtered_captions, image_ids)
        faiss.write_index(index, faiss_index_path)
        with open(retrieved_caps_path, 'w') as f:
            json.dump(retrieved_caps, f)

if __name__ == '__main__':
    main()
