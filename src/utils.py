from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

import jieba

def preprocess_chinese(text):
    """中文分词处理"""
    text = text.replace(" ", "").replace("\n", "")
    return " ".join(jieba.lcut(text))

def load_flickr8k_captions(caption_path):
    captions = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            img_id = parts[0].split('#')[0]
            caption = preprocess_chinese(parts[1])
            captions.append({'image_id': img_id, 'caption': caption})
    return captions
    
def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features, tokenizer, max_caption_length=25):
        self.df = df.reset_index(drop=True)
        self.features = features  # 这里是 numpy ndarray，不是 h5py
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row['caption']
        input_ids = self.tokenizer.EncodeAsIds(caption)
        input_ids = input_ids[:self.max_caption_length]
        input_ids += [self.tokenizer.pad_id()] * (self.max_caption_length - len(input_ids))

        feature = self.features[row['feature_idx']]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "pixel_values": torch.tensor(feature, dtype=torch.float32),
        }


def load_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]

        # ✅ 添加健壮性检查
        if caps_path is not None:
            cocoid_str = str(item['cocoid'])
            if cocoid_str not in retrieved_caps:
                continue  # ⚠️ 跳过缺失检索结果的图像
            caps = retrieved_caps[cocoid_str]
        else:
            caps = None

        samples = []
        for sentence in item['sentences']:
            samples.append({
                'file_name': file_name,
                'cocoid': str(item['cocoid']),
                'caps': caps,
                'text': ' '.join(sentence['tokens'])
            })
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples

    return data


def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        image = {
            'file_name': file_name,
            'caps': caps,
            'image_id': str(item['cocoid'])
        }
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)
    return data      
