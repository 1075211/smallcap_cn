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
    
def get_nns(captions, images, k=15):
    """
    使用FAISS构建索引并检索最近邻
    
    参数:
        captions: 编码后的描述特征 (n_captions, dim)
        images: 编码后的图像特征 (n_images, dim)
        k: 检索数量
    
    返回:
        index: FAISS索引对象
        I: 最近邻索引矩阵 (n_images, k)
    """
    import faiss
    import numpy as np
    
    # 转换为float32类型
    xb = captions.astype(np.float32)  # 描述特征库
    xq = images.astype(np.float32)    # 查询图像特征
    
    # 归一化
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)
    
    # 构建索引
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # 使用内积相似度
    index.add(xb)
    
    # 搜索最近邻
    D, I = index.search(xq, k)  # D: 距离, I: 索引
    
    return index, I
    
def main():
    # 数据路径
    feature_dir = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/FeatureData/pyresnet152-pool5osl2"
    caption_path = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/TextData/seg.flickr8kzhbJanbosontrain.caption.txt"
    
    print('Loading Flickr8k-CN data')
    image_ids, features, captions = load_flickr8k_data(feature_dir, caption_path)
    
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载CLIP模型
    clip_model, _ = clip.load("RN50x64", device=device)
    
    # 1. 过滤过长的描述
    print('Filtering long captions')
    filtered_image_ids, filtered_captions = filter_captions(captions)
    
    # 2. 编码描述
    print('Encoding captions')
    encoded_captions = encode_captions(filtered_captions, clip_model, device)
    
    # 3. 使用预提取图像特征
    print('Using pre-computed image features')
    encoded_features = features  # 直接使用预提取特征
    
    # 4. 构建FAISS索引并检索
    print('Building FAISS index and retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_features)
    
    # 保存结果
    output_dir = "/kaggle/working/datastore"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存索引
    faiss.write_index(index, os.path.join(output_dir, "flickr8k_index"))
    
    # 保存检索结果
    retrieved_caps = filter_nns(nns, filtered_image_ids, filtered_captions, image_ids)
    with open(os.path.join(output_dir, "retrieved_caps.json"), 'w') as f:
        json.dump(retrieved_caps, f)

if __name__ == '__main__':
    main()
