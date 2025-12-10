import os
import glob
import numpy as np
from PIL import Image

def load_dataset_from_directory(dataset_path):
    """
    [功能]: 从文件夹里读取所有图片。
    
    [输入]: dataset_path (文件夹在哪里)
    [输出]: 
        - image_paths: 图片文件的路径列表
        - labels: 图片对应的数字标签
        - class_names: 类别名字 (比如 ['cat', 'dog'])
    """
    # 1. 检查文件夹存不存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
    image_paths = []
    labels = []
    class_names = []
    
    # 2. 扫描文件夹，把子文件夹的名字当作类别名
    # os.listdir 列出所有文件，我们只选文件夹
    entries = os.listdir(dataset_path)
    # 排序一下，保证每次运行顺序都一样
    class_names = sorted([entry for entry in entries if os.path.isdir(os.path.join(dataset_path, entry))])
    
    # 给每个类别编个号，比如 cat=0, dog=1
    class_to_index = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    # 打印一下找到了几个类别
    print(f"[Info] Found {len(class_names)} classes: {class_names}")
    
    # 3. 开始遍历每个类别，找里面的图片
    # 支持这些格式的图片
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        class_idx = class_to_index[class_name]
        
        # 在这个类别的文件夹里找图片
        for file_pattern in supported_formats:
            # 拼凑出完整路径，比如 data/cat/*.jpg
            search_path = os.path.join(class_dir, file_pattern)
            # glob 用来真正查找文件
            found_files = glob.glob(search_path)
            
            # 把找到的图片加到列表里
            for file_path in found_files:
                image_paths.append(file_path)
                labels.append(class_idx)
                
    # 把列表转换成 numpy 数组，方便后面计算
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    print(f"[Info] Loaded {len(image_paths)} images in total.")
    return image_paths, labels, class_names

def load_image(image_path):
    """
    [功能]: 用 PIL 库打开一张单独的图片。
    """
    try:
        img = Image.open(image_path)
        # 必须转成 RGB (彩色模式)，不然有的黑白图后面会报错
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        # 如果出错，打印简单的英文提示
        print(f"[Error] Cannot load image {image_path}: {e}")
        return None

def validate_dataset(dataset_path):
    """
    [功能]: 检查一下数据集是不是空的，统计一下信息。
    """
    if not os.path.exists(dataset_path):
        return {"valid": False, "error": "Path does not exist"}
    
    try:
        # 试着加载一下数据
        image_paths, labels, class_names = load_dataset_from_directory(dataset_path)
        
        # 把统计结果存到字典里
        stats = {
            "valid": True,
            "total_images": len(image_paths),
            "num_classes": len(class_names),
            "classes": class_names,
            "class_distribution": get_class_distribution(labels, class_names)
        }
        
        # 如果一张图都没找到，那就是无效的
        if stats["total_images"] == 0:
            stats["valid"] = False
            stats["error"] = "No images found in the directory"
            
        return stats
    except Exception as e:
        return {"valid": False, "error": str(e)}

def get_class_distribution(labels, class_names=None):
    """
    [功能]: 算算每个类别到底有多少张图。
    """
    # np.unique 可以帮我们计数
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    # 如果给了类别名字，就把 key 从数字 (0, 1) 换成名字 ('cat', 'dog')
    if class_names:
        readable_dist = {}
        for idx, count in distribution.items():
            if idx < len(class_names):
                name = class_names[idx]
                readable_dist[name] = int(count)
        return readable_dist
        
    return distribution