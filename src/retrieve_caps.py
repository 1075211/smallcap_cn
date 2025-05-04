import json
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_coco_data(train_json, val_json):
    def load_annotations(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        anns = data['annotations']
        return id_to_filename, anns

    images = []
    captions = []

    train_id_to_file, train_anns = load_annotations(train_json)
    val_id_to_file, val_anns = load_annotations(val_json)

    def process(anns, id_to_file):
        for ann in anns:
            image_id = ann['image_id']
            caption = ann['caption']
            captions.append({'image_id': image_id, 'caption': caption})
        for image_id, file_name in id_to_file.items():
            images.append({'image_id': image_id, 'file_name': file_name})

    process(train_anns, train_id_to_file)
    process(val_anns, val_id_to_file)

    return images, captions

def filter_captions(data):
    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        encodings += tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np', padding=True)['input_ids'].tolist()
    
    filtered_image_ids, filtered_captions = [], []
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions

def encode_captions(captions, model, device):
    bs = 256
    encoded_captions = []
    for idx in tqdm(range(0, len(captions), bs), desc="Encoding captions"):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx+bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())
    return np.concatenate(encoded_captions)

def encode_images(images, image_path, model, feature_extractor, device):
    image_ids = [i['image_id'] for i in images]
    bs = 64
    image_features = []
    valid_image_ids = []
    for idx in tqdm(range(0, len(images), bs), desc="Encoding images"):
        batch = images[idx:idx+bs]
        image_input = []
        batch_ids = []
        for i in batch:
            img_path = os.path.join(image_path, i['file_name'])
            if not os.path.exists(img_path):
                print(f"Skipping missing image {i['file_name']}")
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                image_input.append(feature_extractor(img))
                batch_ids.append(i['image_id'])
            except Exception as e:
                print(f"Error loading image {i['file_name']}: {e}")
        if image_input:
            with torch.no_grad():
                image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())
            valid_image_ids.extend(batch_ids)
    return valid_image_ids, np.concatenate(image_features)

def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)
    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions

def main():
    train_json = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
    val_json = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'
    root_image_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017/'

    print('Loading COCO 2017 data')
    images, captions = load_coco_data(train_json, val_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load("RN50x64", device=device)

    os.makedirs("/kaggle/working/datastore", exist_ok=True)
    captions_path = "/kaggle/working/datastore/filtered_captions.json"
    encoded_captions_path = "/kaggle/working/datastore/encoded_captions.npy"
    encoded_images_path = "/kaggle/working/datastore/encoded_images.npy"
    xq_image_ids_path = "/kaggle/working/datastore/xq_image_ids.json"
    faiss_index_path = "/kaggle/working/datastore/coco_index"
    retrieved_caps_path = "/kaggle/working/retrieved_caps_resnet50x64.json"

    # Step 1: Filter captions
    if os.path.exists(captions_path):
        print('Loading filtered captions')
        xb_image_ids, captions = json.load(open(captions_path))
    else:
        print('Filtering short captions')
        xb_image_ids, captions = filter_captions(captions)
        json.dump([xb_image_ids, captions], open(captions_path, 'w'))

    # Step 2: Encode captions
    if os.path.exists(encoded_captions_path):
        print('Loading encoded captions')
        encoded_captions = np.load(encoded_captions_path)
    else:
        print('Encoding captions')
        encoded_captions = encode_captions(captions, clip_model, device)
        np.save(encoded_captions_path, encoded_captions)

    # Step 3: Encode images
    if os.path.exists(encoded_images_path) and os.path.exists(xq_image_ids_path):
        print('Loading encoded images')
        encoded_images = np.load(encoded_images_path)
        xq_image_ids = json.load(open(xq_image_ids_path))
    else:
        print('Encoding images')
        xq_image_ids, encoded_images = encode_images(images, root_image_dir, clip_model, feature_extractor, device)
        np.save(encoded_images_path, encoded_images)
        json.dump(xq_image_ids, open(xq_image_ids_path, 'w'))

    # Step 4: Build or load index
    if os.path.exists(retrieved_caps_path):
        print('FAISS index and retrieval already done. Skipping.')
    else:
        print('Building FAISS index and retrieving neighbors')
        index, nns = get_nns(encoded_captions, encoded_images)
        retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)
        faiss.write_index(index, faiss_index_path)
        json.dump(retrieved_caps, open(retrieved_caps_path, 'w'))

if __name__ == '__main__':
    main()
