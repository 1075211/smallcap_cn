import json
from tqdm import tqdm
import jieba
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import psutil
import pandas as pd  # 新增WebQA数据处理

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------- 新增函数：WebQA数据处理 -------------------
def load_webqa_data(webqa_path):
    """加载WebQA数据集并构建检索库"""
    print("加载WebQA数据...")
    df = pd.read_csv(webqa_path)
    
    # 提取问题和证据文本作为检索键
    questions = df["question"].fillna("").tolist()
    evidences = df["evidence"].fillna("").tolist()
    answers = df["answer"].fillna("").tolist()
    
    # 组合问题和证据作为检索内容
    corpus = [f"问题：{q}；证据：{e}" for q, e in zip(questions, evidences)]
    return corpus, answers

def encode_webqa(corpus, model, device, batch_size=128):
    """编码WebQA文本"""
    encoded = []
    model = model.to(device)
    
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding WebQA"):
        batch = corpus[i:i+batch_size]
        with torch.no_grad():
            inputs = clip.tokenize(batch, truncate=True).to(device)
            batch_encoded = model.encode_text(inputs).float().cpu().numpy()
            encoded.append(batch_encoded)
    
    return np.concatenate(encoded)

# ------------------- 主函数修改 -------------------
def main():
    try:
        # 数据路径
        feature_dir = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/FeatureData/pyresnet152-pool5osl2"
        caption_path = "/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/TextData/seg.flickr8kzhbJanbosontrain.caption.txt"
        webqa_path = "/kaggle/input/webqadata/webqa_data.csv"  # WebQA路径
        
        # 1. 加载Flickr8k-CN数据
        print('1. 加载Flickr8k-CN数据')
        image_ids, features, captions = load_flickr8k_data(feature_dir, caption_path)
        
        # 2. 加载WebQA数据
        print('2. 加载WebQA数据')
        webqa_corpus, webqa_answers = load_webqa_data(webqa_path)
        
        # 设备设置
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载CLIP模型
        print('3. 加载CLIP模型')
        with torch.no_grad():
            clip_model, _ = clip.load("RN50x64", device=device, jit=False)
            clip_model.eval()
        
        # 4. 编码Flickr8k描述
        print('4. 编码Flickr8k描述')
        encoded_captions = encode_captions([c['caption'] for c in captions], clip_model, device)
        
        # 5. 编码WebQA数据
        print('5. 编码WebQA数据')
        encoded_webqa = encode_webqa(webqa_corpus, clip_model, device)
        
        # 6. 合并检索库（Flickr8k + WebQA）
        print('6. 合并检索库')
        combined_features = np.vstack([encoded_captions, encoded_webqa])
        combined_labels = [{"type": "caption", "data": c} for c in captions] + \
                         [{"type": "webqa", "data": a} for a in webqa_answers]
        
        # 7. 构建FAISS索引
        print('7. 构建FAISS索引')
        index = faiss.IndexFlatIP(combined_features.shape[1])
        index.add(combined_features)
        
        # 8. 示例：用关键词检索（可从图像生成或用户输入）
        test_keywords = "泰山 海拔"  # 示例关键词
        print(f'8. 检索关键词: "{test_keywords}"')
        
        # 编码关键词
        with torch.no_grad():
            keyword_embedding = clip_model.encode_text(clip.tokenize([test_keywords]).to(device)).cpu().numpy()
        
        # 检索Top-5相关结果
        D, I = index.search(keyword_embedding, k=5)
        
        # 打印结果
        print("\n检索结果：")
        for idx in I[0]:
            item = combined_labels[idx]
            if item["type"] == "caption":
                print(f"[图像描述] {item['data']['caption']}")
            else:
                print(f"[WebQA答案] {item['data']}")
        
        # 保存索引
        output_dir = "/kaggle/working/datastore"
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(output_dir, "combined_index"))
        
        print("处理完成！")
        
    except Exception as e:
        print(f"错误发生：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
