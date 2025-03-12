import os, sys, re
from pathlib import Path
sys.path.append(str(Path().resolve()))
from PIL import Image as PILImage
from datasets import Dataset, Features, ClassLabel, DatasetDict, load_from_disk, Image as HFImage
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from collections import Counter
from src.chatbot import pathtree

del_name = ["copilot+ x plus", "gaming", " - chỉ có tại cellphones", " - nhập khẩu chính hãng", "2-in-1"]
spe_name = ["laptop hp 14", "laptop hp 14s", "laptop hp 15", "laptop hp 15s", "laptop hp", "laptop lenovo v"]
acc_name = [
    "aspire", "nitro", "swift", "expertbook", "flow", "strix", "zephyrus", "tuf", "vivobook",
    "zenbook", "vostro", "xps", "gigabyte", "dragonfly", "elitebook", "envy", "omen",
    "pavilion", "probook", "spectre", "victus", "ideapad", "legion", "loq", "thinkbook",
    "thinkpad", "yoga", "gram", "gf63", "bravo", "cyborg", "katana", "modern", "prestige",
    "raider", "stealth", "sword", "thin", "titan", "vector"
]

PATTERN_HYPHEN_WORDS = re.compile(r'\b\w*-\w*\b')
PATTERN_ACER_CLEAN = re.compile(r'\s+(?:(?!\d+\.jpg)\S*[.-]\S*\s+)+(?=\d+\.jpg)')
PATTERN_DELL_REPLACE = re.compile(r'(\b\d{4}\b)\s+\S+(?=\s+\d+\.jpg)')
PATTERN_HP_REMOVE = re.compile(r'-\S+(?:\s+\S+)*(?=\s+\d+\.jpg)')
PATTERN_HP_REPLACE = re.compile(r'(\b\d+\s+\w+\s+)\w+(?=\s+\d+\.jpg\b)')
PATTERN_LENOVO_REMOVE = re.compile(r'\s+(?P<tok1>(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+)\s+(?P<tok2>(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+)(?=\s+\d+\.jpg\b)')
PATTERN_LENOVO_REPL = re.compile(r'(\s+)(\S+)(?=\s+\d+\.jpg\b)')
PATTERN_LG_REMOVE = re.compile(r'\b(?=\S*-\S*)(?=\S*\.\S*)\S+\b')
PATTERN_ALPHA = re.compile(r'[A-Za-z]')
PATTERN_DIGIT = re.compile(r'\d')
MODEL_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(name) for name in acc_name) + r')\b')

def rename():
    for folder in os.listdir(pathtree().get('cellphones')):
        folder_path = pathtree().get(folder)
        file_path = os.listdir(folder_path)
        for filename in file_path:
            old_name = os.path.join(folder_path, filename)
            new_name = old_name.lower()
            for dname in del_name:
                new_name = new_name.replace(dname, "")
            os.rename(old_name, new_name.replace("  ", " "))

def remove_junk(text, brand):
    if brand in ["asus", "gigabyte", "msi"]:
        text = PATTERN_HYPHEN_WORDS.sub(' ', text)
    elif brand == "acer":
        text = PATTERN_ACER_CLEAN.sub(' ', text)
    elif brand == "dell":
        text = PATTERN_DELL_REPLACE.sub(r'\1', text)
        text = PATTERN_HYPHEN_WORDS.sub(' ', text)
    elif brand == "hp":
        text = PATTERN_HP_REMOVE.sub('', text).strip()
        text = PATTERN_HP_REPLACE.sub(r'\1', text)
    elif brand == "lenovo":
        text = PATTERN_LENOVO_REMOVE.sub(' ', text)
        def repl(match):
            whitespace = match.group(1)
            token = match.group(2)
            if len(token) >= 6 and PATTERN_ALPHA.search(token) and PATTERN_DIGIT.search(token):
                return whitespace
            return match.group(0)
        text = PATTERN_LENOVO_REPL.sub(repl, text)
    elif brand == "lg":
        text = PATTERN_LG_REMOVE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def best_size(dataset, threshold: float = 10):
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    main_bar = tqdm(total=3, desc="Processing image sizes", bar_format=bar_format)

    size_counter = Counter()
    for example in dataset:
        image = example['image']
        if isinstance(image, PILImage.Image):
            size = image.size
            size_counter[size] += 1
    main_bar.update(1)
    
    valid_size = []
    for size, _ in size_counter.items():
        size_diff_threshold = (1 - (min(size) / max(size))) * 100
        if max(size) == min(size) or size_diff_threshold < threshold:
            valid_size.append(size)

    if not valid_size:
        main_bar.close()
        raise ValueError("No valid image sizes found that meet the criteria")
    main_bar.update(1)
    
    square_sizes = [size for size in valid_size if max(size) == min(size)]
    if square_sizes:
        target_size = min(square_sizes, key=lambda sizes: sizes[0])
    else:
        target_size = max(valid_size, key=lambda size: size_counter[size])
        
    main_bar.set_description(f"Processing complete. Target size: {target_size[0]}x{target_size[1]}")
    main_bar.update(1)
    main_bar.close() 

    return target_size, valid_size

def resize_image(example, target_size: tuple):
    example['image'] = example['image'].resize(target_size)
    return example

def transform_image_mode(example):
    image = example["image"]
    try:
        if not isinstance(image, PILImage.Image):
            print(f"Converting to PIL Image: {type(image)}")
            image = PILImage.open(image)
        channels = len(image.getbands())
        if channels == 4:
            print(f"Converting {image.mode} (4 channels) to RGB")
            image = image.convert('RGB')
        elif channels != 3: 
            print(f"Unusual mode {image.mode} ({channels} channels), converting to RGB")
            image = image.convert('RGB')
            
        return {"image": image, "label": example["label"]}
    
    except Exception as error:
        print(f"Error converting image: {error}")
        default_image = PILImage.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        return {"image": default_image, "label": example["label"]}

def data_prep(resize: bool = True):
    cpath = pathtree().get("cellphones")
    data = {"image": [], "label": []}
    for folder in os.listdir(cpath):
        folder_path = pathtree().get(folder)
        file_path = os.listdir(folder_path)
        for filename in file_path:
            clean_name = re.sub(r'\b\d+\.jpg\b', '', remove_junk(filename, folder)).strip()
            if any(substring in clean_name for substring in spe_name) and "hp" in clean_name:
                label = "laptop hp series"
            elif any(substring in clean_name for substring in spe_name) and "lenovo" in clean_name:
                label = "laptop lenovo v series"
            else:
                match = MODEL_PATTERN.search(clean_name)
                label = clean_name[:match.end()] if match else clean_name
            data["label"].append(label)
            data["image"].append(os.path.join(folder_path, filename))

    dataset = Dataset.from_dict(data)
    labels = sorted(list(set(data["label"])))
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=labels)
    })
    dataset = dataset.cast(features, num_proc=8)

    if resize:
        target_size, valid_sizes = best_size(dataset)
        dataset = dataset.filter(
            lambda example: example["image"].size in valid_sizes,
            num_proc=8,
            desc="Filtering images with valid sizes",
        )
        dataset = dataset.map(
            lambda example: resize_image(example, target_size),
            num_proc=8,
            desc=f"Resizing images to {target_size[0]}x{target_size[1]}",
        )

    dataset = dataset.map(
        transform_image_mode,
        desc="Converting RGBA to RGB",
        num_proc=8
        )
    return dataset

def build_dataset(save_path: str) -> DatasetDict:
    if os.path.exists(os.path.join(save_path, "dataset_dict.json")):
        return load_from_disk(save_path)
    
    dataset = data_prep()
    original_labels = dataset.features["label"].names
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=original_labels)
    })
    
    with tqdm(total=len(dataset), desc="Extracting Examples") as main_bar:    
        examples = [{"image": ex["image"], "label": ex["label"]} for ex in dataset]
        main_bar.update(len(dataset))

    train_data, test_data = train_test_split(examples, test_size=0.2, random_state=42)

    with tqdm(total=2, desc="Creating dataset") as pbar:
        pbar.set_description("Processing train set")
        hf_train_dataset = Dataset.from_dict({
            "image": [ex["image"] for ex in train_data],
            "label": [ex["label"] for ex in train_data]
        }, features=features)
        pbar.update(1)

        pbar.set_description("Processing test set")
        hf_test_dataset = Dataset.from_dict({
            "image": [ex["image"] for ex in test_data],
            "label": [ex["label"] for ex in test_data]
        }, features=features)
        pbar.update(1)
    processed_dataset = DatasetDict({"train": hf_train_dataset, "test": hf_test_dataset})
    processed_dataset.save_to_disk(save_path)
    
    return processed_dataset

def check_images(dataset_dict: DatasetDict, return_info: bool = False):
    """
    Kiểm tra xem DatasetDict chỉ chứa 2 kích thước ảnh và 2 loại kênh màu hay không.
    
    Args:
        dataset_dict: DatasetDict chứa các split (train, test, v.v.).
    
    Returns:
        tuple: (bool, bool, set, set) - 
            - Có đúng 1 kích thước không, 
            - Có đúng 1 loại kênh không, 
            - Tập hợp kích thước, 
            - Tập hợp số kênh.
    """
    all_sizes = set()
    all_channels = set() 
    for split_name, dataset in dataset_dict.items():
        print(f"Checking split: {split_name} ({len(dataset)} samples)")
        
        for i, example in enumerate(dataset):
            image = example["image"]
            
            try:
                if not isinstance(image, PILImage.Image):
                    print(f"Split {split_name}, Sample {i}: Converting to PIL Image")
                    image = PILImage.fromarray(np.array(image))
                
                size = image.size  
                channels = len(image.getbands())  
                
                all_sizes.add(size)
                all_channels.add(channels)

                if len(all_sizes) > 1:
                    print(f"Split {split_name}, Sample {i}: Found more than 1 size - {all_sizes}")
                if len(all_channels) > 1:
                    print(f"Split {split_name}, Sample {i}: Found more than 1 channel type - {all_channels}")
            
            except Exception as e:
                print(f"Split {split_name}, Sample {i}: Error - {e}")
    exactly_size = len(all_sizes) == 1
    exactly_channel = len(all_channels) == 1
    
    print("\nSummary:")
    print(f"Unique sizes found: {all_sizes}")
    print(f"Has exactly 1 size: {exactly_size}")
    print(f"Unique channel types found: {all_channels} chanels")
    print(f"Has exactly 1 channel type: {exactly_channel}")
    
    if return_info:
        return exactly_size, exactly_channel, all_sizes, all_channels