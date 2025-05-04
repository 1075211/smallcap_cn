import os
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.opt import ThisOPTConfig, ThisOPTForCausalLM
from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.vision_encoder_decoder import SmallCap, SmallCapConfig
import json
import h5py
import torch
import argparse
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoConfig, AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM,
    Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel, EncoderDecoderModel
)
import pandas as pd
import numpy as np
os.environ["WANDB_DISABLED"] = "true"
from src.utils import *

# 常量定义
CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    """准备输入字符串和标签"""
    padding = not is_test
    truncation = not is_test

    prefix = template if template is not None else SIMPLE_PREFIX

    if retrieved_caps is not None and k is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = prefix.replace('||', infix)

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

    return input_ids if is_test else (input_ids, label_ids)

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.max_target_length = max_caption_length
        self.rag = rag
        self.k = k

        # 加载特征文件
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")
        self.features = h5py.File(features_path, 'r')

        # 处理模板
        if rag:
            if template_path and os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    self.template = f.read().strip() + ' '
            else:
                print(f"Warning: Template file not found at {template_path}, using default prefix")
                self.template = SIMPLE_PREFIX
        self.max_target_length = (
            max_caption_length  # target caption
            + max_caption_length * k  # retrieved captions
            + len(tokenizer.encode(self.template))  # template
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        text = item['text']
        caps = item['caps'] if self.rag else None
        
        if self.rag:
            input_ids, labels = prep_strings(
                text, self.tokenizer, 
                template=self.template,
                retrieved_caps=caps,
                k=self.k,
                max_length=self.max_target_length
            )
        else:
            input_ids, labels = prep_strings(
                text, self.tokenizer,
                max_length=self.max_target_length
            )

        # 加载预计算特征
        encoder_outputs = self.features[item['cocoid']][()]
        
        return {
            "encoder_outputs": torch.tensor(encoder_outputs),
            "decoder_input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }

def load_data_for_training(annot_path, caps_path=None):
    """加载COCO官方格式的训练数据"""
    print(f"Loading annotations from: {annot_path}")
    with open(annot_path) as f:
        annotations = json.load(f)

    # 构建 image_id -> captions 列表映射
    id2captions = {}
    for ann in annotations['annotations']:
        cocoid = str(ann['image_id'])
        id2captions.setdefault(cocoid, []).append(ann['caption'])

    # 加载检索增强字幕
    retrieved_caps = None
    if caps_path and os.path.exists(caps_path):
        with open(caps_path) as f:
            retrieved_caps = json.load(f)
        print(f"Loaded {len(retrieved_caps)} retrieved captions")

    # 合并为训练/验证结构
    data = {'train': [], 'val': []}
    for item in annotations['images']:
        cocoid = str(item['id'])
        file_name = item['file_name']
        caps = retrieved_caps.get(cocoid, None) if retrieved_caps else None

        # 检查 image 是否有 caption
        if cocoid not in id2captions:
            continue

        # 根据文件名判断所属数据集
        if 'train' in file_name:
            split = 'train'
        elif 'val' in file_name:
            split = 'val'
        else:
            continue

        # 每条 caption 单独作为一条训练样本
        for caption in id2captions[cocoid]:
            data[split].append({
                'cocoid': cocoid,
                'file_name': file_name,
                'text': caption,
                'caps': caps
            })

    print(f"Loaded {len(data['train'])} training samples and {len(data['val'])} validation samples")
    return data

def get_model_and_auxiliaries(args):
    """初始化模型和相关组件"""
    # 注册自定义模型
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    # 初始化组件
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]
    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    # 初始化模型
    model = SmallCap.from_encoder_decoder_pretrained(
        args.encoder_name, 
        args.decoder_name, 
        cross_attention_reduce_factor=cross_attention_reduce_factor
    )
    
    # 配置模型
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = CAPTION_LENGTH
    model.config.rag = not args.disable_rag

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder

    # 冻结参数
    for param in model.encoder.parameters():
        param.requires_grad = False

    if not args.train_decoder:
        for name, param in model.decoder.named_parameters():
            if 'xglm' in args.decoder_name or 'opt' in args.decoder_name:
                if 'encoder_attn' not in name:
                    param.requires_grad = False
            else:
                if 'crossattention' not in name:
                    param.requires_grad = False

    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training model with {trainable_params} trainable parameters')

    return model, tokenizer, feature_extractor

def main(args):
    """主训练函数"""
    # 初始化模型
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    
    # 加载数据
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    # 设置输出目录
    model_type = 'norag' if args.disable_rag else 'rag'
    if args.ablation_visual:
        output_dir = f"{model_type}_{args.attention_size}M_{args.decoder_name}_ablation"
    else:
        output_dir = f"{model_type}_{args.attention_size}M_{args.decoder_name}"
    
    output_dir = os.path.join(args.experiments_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs,
        logging_strategy="epoch",
        prediction_loss_only=True,
        remove_unused_columns=False,
        report_to="none"
    )

    # 初始化Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=feature_extractor,
    )

    # 开始训练
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    
    # 数据路径参数
    parser.add_argument("--features_dir", type=str, default="/kaggle/working/features/", 
                       help="Directory with cached image features")
    parser.add_argument("--annotations_path", type=str, 
                       default="/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json", 
                       help="Path to annotations JSON file")
    parser.add_argument("--captions_path", type=str, 
                       default="/kaggle/working/retrieved_caps_resnet50x64.json", 
                       help="Path to retrieved captions JSON file")
    parser.add_argument("--template_path", type=str, 
                       default="/kaggle/working/smallcap/src/template.txt", 
                       help="Path to template text file")
    parser.add_argument("--experiments_dir", type=str, 
                       default="/kaggle/working/", 
                       help="Directory to save trained models")

    # 模型参数
    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", 
                       help="Name of encoder model")
    parser.add_argument("--decoder_name", type=str, default="gpt2", 
                       help="Name of decoder model")
    parser.add_argument("--attention_size", type=float, default=7, 
                       help="Size of cross attention parameters")
    parser.add_argument("--train_decoder", action="store_true", 
                       help="Whether to train decoder parameters")
    parser.add_argument("--disable_rag", action="store_true", 
                       help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, 
                       help="Number of retrieved captions to use")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", 
                       help="Visual encoder used for retrieval")
    parser.add_argument("--ablation_visual", action="store_true", 
                       help="Whether to blank visual features")

    # 训练参数
    parser.add_argument("--n_epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, 
                       help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, 
                       help="Gradient accumulation steps")

    args = parser.parse_args()
    
    # 打印配置
    print("\nTraining configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print()

    main(args)
