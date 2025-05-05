import pandas as pd
import numpy as np
import os
import argparse
import jieba  # 中文分词
os.environ["WANDB_DISABLED"] = "true"

from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel, EncoderDecoderModel

from src.vision_encoder_decoder import SmallCap, SmallCapConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.opt import ThisOPTConfig, ThisOPTForCausalLM

from src.utils import *

# 全局参数
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '[PAD]'  # 中文常用Pad Token
EOS_TOKEN = '[SEP]'  # 中文常用结束符
CAPTION_LENGTH = 25

def load_flickr8k_features(feature_path, id_path, shape_path):
    """加载Flickr8k-CN预提取特征"""
    with open(shape_path, 'r') as f:
        shape = tuple(map(int, f.read().strip().split(',')))
    features = np.fromfile(feature_path, dtype=np.float32).reshape(shape)
    with open(id_path, 'r') as f:
        image_ids = [line.strip() for line in f]
    return features, image_ids

def load_flickr8k_captions(caption_path):
    """加载中文描述并分词"""
    captions = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            img_id = parts[0].split('#')[0]
            caption = ' '.join(jieba.cut(parts[1]))  # 中文分词
            captions.append({'image_id': img_id, 'caption': caption})
    return captions



def get_model_and_auxiliaries(args):
    # 加载中文Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  # 或 "uer/gpt2-chinese-cluecorpussmall"
    tokenizer.add_special_tokens({
        'pad_token': PAD_TOKEN,
        'eos_token': EOS_TOKEN
    })
    
    # 初始化Mengzi配置
    config = MengziGPTConfig(
        vocab_size=len(tokenizer),
        n_positions=CAPTION_LENGTH,
        n_embd=768,  # 与CLIP-ViT输出维度对齐
        n_layer=12,   # 可调整层数减少计算量
        n_head=12,
        cross_attention_reduce_factor=PARAMS2REDUCE_FACTOR[args.attention_size],
    )
    
    # 构建模型
    encoder = CLIPModel.from_pretrained(args.encoder_name)
    decoder = MengziGPTLMHeadModel(config)
    
    # 初始化时指定decoder_name为"mengzi-gpt"
    model = SmallCap.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path="openai/clip-vit-base-patch32",
        decoder_pretrained_model_name_or_path="langboat/mengzi-gpt-neo",  # 自动识别为mengzi-gpt类型
        cross_attention_reduce_factor=4
    )
    
    # 冻结编码器
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # 投影层（CLIP-ViT输出768维 → Mengzi输入768维）
    model.proj = nn.Linear(encoder.config.projection_dim, config.n_embd)
    
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    return model, tokenizer, None

def get_data(tokenizer, max_length, args):
    """加载Flickr8k-CN数据"""
    # 加载特征和描述
    features, image_ids = load_flickr8k_features(
        os.path.join(args.features_dir, 'feature.bin'),
        os.path.join(args.features_dir, 'id.txt'),
        os.path.join(args.features_dir, 'shape.txt')
    )
    captions = load_flickr8k_captions(args.captions_path)

    # 构建DataFrame
    data = {'image_id': [], 'caption': [], 'feature_idx': []}
    for idx, img_id in enumerate(image_ids):
        img_captions = [c['caption'] for c in captions if c['image_id'] == img_id]
        for cap in img_captions:
            data['image_id'].append(img_id)
            data['caption'].append(cap)
            data['feature_idx'].append(idx)
    df = pd.DataFrame(data)

    # 构建Dataset
    if args.ablation_visual:
        dataset = AblationFeaturesDataset(
            df=df,
            features=features,
            tokenizer=tokenizer,
            max_caption_length=max_length
        )
    else:
        dataset = TrainDataset(
            df=df,
            features=features,
            tokenizer=tokenizer,
            max_caption_length=max_length
        )
    return dataset

def main(args):
    # 初始化模型和数据
    model, tokenizer, _ = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, CAPTION_LENGTH, args)

    # 训练配置
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.experiments_dir, f"flickr8k_cn_{args.decoder_name}"),
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=True,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--features_dir", default="/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/FeatureData/pyresnet152-pool5osl2")
    parser.add_argument("--captions_path", default="/kaggle/input/flickr8k-cn-wang/flickr8kzhbJanbosontrain/TextData/seg.flickr8kzhbJanbosontrain.caption.txt")
    parser.add_argument("--experiments_dir", default="/kaggle/working")

    # 模型参数
    parser.add_argument("--encoder_name", default="openai/clip-vit-base-patch32")  # 实际不使用，仅占位
    parser.add_argument("--decoder_name", default="bert-base-chinese")  # 中文解码器
    parser.add_argument("--attention_size", type=float, default=7)

    # 训练参数
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
