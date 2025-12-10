import os
import glob
import numpy as np
from PIL import Image

def load_dataset_from_directory(dataset_path):
    """
    [功能]: 从文件夹里读取所有图片。
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
    image_paths = []
    labels = []
    class_names = []
    
    # 用集合(Set)来记录已经加载过的文件，防止Windows下大小写重复加载
    seen_files = set()
    
    entries = os.listdir(dataset_path)
    class_names = sorted([entry for entry in entries if os.path.isdir(os.path.join(dataset_path, entry))])
    
    class_to_index = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    print(f"[Info] Found {len(class_names)} classes: {class_names}")
    
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        class_idx = class_to_index[class_name]
        
        for file_pattern in supported_formats:
            search_path = os.path.join(class_dir, file_pattern)
            found_files = glob.glob(search_path)
            
            for file_path in found_files:
                # 统一路径格式，并检查是否已经加载过
                norm_path = os.path.normpath(file_path)
                
                if norm_path not in seen_files:
                    image_paths.append(norm_path)
                    labels.append(class_idx)
                    seen_files.add(norm_path) # 标记为已加载
                
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"[Info] Loaded {len(image_paths)} images in total.")
    return image_paths, labels, class_names

def load_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"[Error] Cannot load image {image_path}: {e}")
        return None

def validate_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        return {"valid": False, "error": "Path does not exist"}
    
    try:
        image_paths, labels, class_names = load_dataset_from_directory(dataset_path)
        
        stats = {
            "valid": True,
            "total_images": len(image_paths),
            "num_classes": len(class_names),
            "classes": class_names,
            "class_distribution": get_class_distribution(labels, class_names)
        }
        
        if stats["total_images"] == 0:
            stats["valid"] = False
            stats["error"] = "No images found in the directory"
            
        return stats
    except Exception as e:
        return {"valid": False, "error": str(e)}

def get_class_distribution(labels, class_names=None):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    if class_names:
        readable_dist = {}
        for idx, count in distribution.items():
            if idx < len(class_names):
                name = class_names[idx]
                readable_dist[name] = int(count)
        return readable_dist
        
    return distribution