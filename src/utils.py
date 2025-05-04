from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
import os
CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "


def prep_strings(text, tokenizer, template_path=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True

    # 加载模板文件
    if template_path is not None and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template = f.read().strip() + ' '
    else:
        # 如果没有提供模板路径，则使用默认模板
        template = SIMPLE_PREFIX

    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = template

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
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
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)
        # load precomputed features
        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['file_name']  # 获取文件名
        cocoid = str(item['id'])  # 使用 'id' 作为 cocoid
        if caps_path is not None:
            caps = retrieved_caps.get(cocoid, None)  # 获取 captions
        else:
            caps = None
        
        # 获取 captions
        captions = item.get('captions', [])  # 获取 'captions' 字段, 如果没有则为 []
        
        samples = []
        for caption in captions:
            samples.append({'file_name': file_name, 'cocoid': cocoid, 'caps': caps, 'text': ' '.join(caption['tokens'])})

        # 手动划分数据集，基于文件路径判断训练集和验证集
        if 'train2017' in annot_path:  # 如果路径包含 'train2017'，为训练集
            data['train'] += samples
        elif 'val2017' in annot_path:  # 如果路径包含 'val2017'，为验证集
            data['val'] += samples
    
    return data

def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['file_name']  # 直接获取 file_name
        cocoid = str(item['id'])  # 使用 'id' 作为 cocoid
        if caps_path is not None:
            caps = retrieved_caps.get(cocoid, None)  # 获取 captions
        else:
            caps = None
        
        image = {'file_name': file_name, 'caps': caps, 'image_id': cocoid}
        
        # 根据文件路径判断数据集
        if 'test2017' in annot_path:  # 如果路径包含 'test2017'，为测试集
            data['test'].append(image)
        elif 'val2017' in annot_path:  # 如果路径包含 'val2017'，为验证集
            data['val'].append(image)

    return data
