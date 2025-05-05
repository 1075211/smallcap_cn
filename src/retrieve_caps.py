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
import psutil  # 新增内存监控

ImageFile.LOAD_TRUNCATED_IMAGES = True

def print_memory_usage():
    """打印当前内存使用情况"""
    mem = psutil.virtual_memory()
    print(f"内存使用：{mem.used/1024/1024:.2f}MB / {mem.total/1024/1024:.2f}MB ({mem.percent}%)")

def load_flickr8k_data(feature_dir, caption_path):
    """加载Flickr8k-CN预提取特征和描述（优化内存使用）"""
    print_memory_usage()
    
    # 1. 加载图像特征（使用内存映射减少内存占用）
    shape_path = os.path.join(feature_dir, "shape.txt")
    with open(shape_path, 'r') as f:
        shape = tuple(map(int, f.read().strip().split()))
    
    print(f"加载特征文件，形状：{shape}...")
    features = np.memmap(
        os.path.join(feature_dir, "feature.bin"),
        dtype=np.float32,
        mode='r',
        shape=shape
    )
    
    # 2. 加载图像ID
    id_path = os.path.join(feature_dir, "id.txt")
    with open(id_path, 'r') as f:
        image_ids = [line.strip() for line in f]
    
    # 3. 加载中文描述（分批处理大文件）
    captions = []
    print("加载描述文件...")
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing captions"):
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                img_id = parts[0].split('#')[0]
                captions.append({'image_id': img_id, 'caption': parts[1]})
    
    print_memory_usage()
    return image_ids, features, captions

def filter_captions(captions, max_length=25):
    """过滤过长的中文描述（优化分词性能）"""
    print("初始化jieba分词器...")
    jieba.initialize()  # 显式初始化
    
    filtered_image_ids, filtered_captions = [], []
    for item in tqdm(captions, desc="Filtering captions"):
        words = list(jieba.cut(item['caption']))
        if len(words) <= max_length:
            filtered_image_ids.append(item['image_id'])
            filtered_captions.append(item['caption'])
    return filtered_image_ids, filtered_captions

def encode_captions(captions, model, device, batch_size=128):
    """用CLIP编码中文描述（优化内存和性能）"""
    encoded = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
        batch = captions[i:i+batch_size]
        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = clip.tokenize(batch, truncate=True).to(device)
            batch_encoded = model.encode_text(inputs).float().cpu().numpy()
            encoded.append(batch_encoded)
        
        # 每隔10个批次释放内存
        if i % (10*batch_size) == 0:
            torch.cuda.empty_cache()
    
    return np.concatenate(encoded)

def get_nns(captions, images, k=15):
    """使用FAISS构建索引并检索（优化大内存处理）"""
    # 转换为float32类型
    xb = np.ascontiguousarray(captions.astype(np.float32))
    xq = np.ascontiguousarray(images.astype(np.float32))
    
    # 归一化
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)
    
    print("构建FAISS索引...")
    dim = xb.shape[1]
    
    # 对于大数据集，使用IVF索引减少内存
    nlist = min(100, xb.shape[0]//100)  # 聚类中心数
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    if not index.is_trained:
        print("训练聚类中心...")
        index.train(xb)
    
    print("添加向量到索引...")
    batch_size = 10000
    for i in tqdm(range(0, xb.shape[0], batch_size)):
        index.add(xb[i:i+batch_size])
    
    print("搜索最近邻...")
    D, I = index.search(xq, k)
    return index, I

def filter_nns(nns, caption_image_ids, captions, query_image_ids, k=7):
    """过滤检索结果（优化处理速度）"""
    retrieved = {}
    caption_id_map = {cid:i for i,cid in enumerate(caption_image_ids)}
    
    for i, img_id in tqdm(enumerate(query_image_ids), desc="Filtering neighbors"):
        valid_captions = []
        current_caption_idx = caption_id_map.get(img_id, -1)
        
        for nn_idx in nns[i]:
            if nn_idx != current_caption_idx:
                valid_captions.append(captions[nn_idx])
                if len(valid_captions) >= k:
                    break
        retrieved[img_id] = valid_captions
    return retrieved

def main():
    try:
        # 数据路径
        feature_dir = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/FeatureData/pyresnet152-pool5osl2"
        caption_path = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/TextData/seg.flickr8kzhbJanbosontrain.caption.txt"
        
        print('1. 加载Flickr8k-CN数据')
        image_ids, features, captions = load_flickr8k_data(feature_dir, caption_path)
        
        # 设备设置
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载CLIP模型（释放不需要的组件）
        print('2. 加载CLIP模型')
        with torch.no_grad():
            clip_model, _ = clip.load("RN50x64", device=device, jit=False)
            clip_model.eval()
        
        # 1. 过滤过长的描述
        print('3. 过滤过长的描述')
        filtered_image_ids, filtered_captions = filter_captions(captions)
        
        # 2. 编码描述
        print('4. 编码描述')
        encoded_captions = encode_captions(filtered_captions, clip_model, device)
        
        # 3. 使用预提取图像特征
        print('5. 准备图像特征')
        encoded_features = np.array(features)  # 将memmap转为普通数组
        
        # 4. 构建FAISS索引并检索
        print('6. 构建FAISS索引并检索')
        index, nns = get_nns(encoded_captions, encoded_features)
        
        # 保存结果
        output_dir = "/kaggle/working/datastore"
        os.makedirs(output_dir, exist_ok=True)
        
        print('7. 保存索引')
        faiss.write_index(index, os.path.join(output_dir, "flickr8k_index"))
        
        print('8. 保存检索结果')
        retrieved_caps = filter_nns(nns, filtered_image_ids, filtered_captions, image_ids)
        with open(os.path.join(output_dir, "retrieved_caps.json"), 'w') as f:
            json.dump(retrieved_caps, f, ensure_ascii=False, indent=2)
            
        print("处理完成！")
        
    except Exception as e:
        print(f"错误发生：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
