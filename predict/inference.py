"""
图像分类预测推理模块

提供模型加载、图像预处理、单/批量预测和结果格式化功能。
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from tensorflow import keras

# 可选进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_model(model_name: str, weights_path: str) -> keras.Model:
    """
    加载训练好的Keras模型
    
    Args:
        model_name: 模型名称 (VGG16/ResNet50/MobileNetV2)
        weights_path: 模型权重文件路径 (.h5格式)
    
    Returns:
        加载的Keras模型
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    try:
        model = keras.models.load_model(weights_path)
        print(f"✓ 成功加载 {model_name} 模型")
        return model
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")


def preprocess_input_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    预处理单张图像
    
    处理流程：加载图像 → 调整大小 → 归一化 → 添加batch维度
    
    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸，默认(224, 224)
    
    Returns:
        预处理后的图像数组，形状为(1, 224, 224, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    try:
        # 加载并转换为RGB
        img = Image.open(image_path).convert('RGB')
        
        # 调整大小
        img = img.resize(target_size, Image.LANCZOS)
        
        # 转换为数组并归一化
        img_array = np.array(img, dtype='float32') / 255.0
        
        # 添加batch维度
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise Exception(f"图像预处理失败: {str(e)}")


def predict_single_image(
    model: keras.Model,
    image_path: str,
    class_names: List[str],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    预测单张图像的类别
    
    Args:
        model: Keras模型
        image_path: 图像路径
        class_names: 类别名称列表
        top_k: 返回前K个预测结果
    
    Returns:
        包含预测结果的字典：
        - image_path: 图像路径
        - predicted_class: 预测类别
        - confidence: 置信度
        - top_k_predictions: Top-K预测列表
    """
    # 预处理图像
    img_array = preprocess_input_image(image_path)
    
    # 模型预测
    predictions = model.predict(img_array, verbose=0)[0]
    
    # 获取Top-K结果
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_preds = [(class_names[i], float(predictions[i])) for i in top_k_indices]
    
    # 返回结果
    best_idx = top_k_indices[0]
    return {
        'image_path': image_path,
        'predicted_class': class_names[best_idx],
        'confidence': float(predictions[best_idx]),
        'top_k_predictions': top_k_preds
    }


def predict_batch(
    model: keras.Model,
    image_paths: List[str],
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    批量预测多张图像
    
    Args:
        model: Keras模型
        image_paths: 图像路径列表
        class_names: 类别名称列表
    
    Returns:
        预测结果列表
    """
    results = []
    
    # 创建进度条
    iterator = tqdm(image_paths, desc="预测中", unit="张") if TQDM_AVAILABLE else image_paths
    if not TQDM_AVAILABLE:
        print(f"正在处理 {len(image_paths)} 张图像...")
    
    # 处理每张图像
    for i, image_path in enumerate(iterator):
        try:
            result = predict_single_image(model, image_path, class_names, top_k=1)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  处理失败 {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
        
        # 进度提示（无tqdm时）
        if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(image_paths)}")
    
    if not TQDM_AVAILABLE:
        print(f"✓ 完成: {len(results)} 张图像")
    
    return results


def format_predictions(prediction_dict: Dict[str, Any]) -> str:
    """
    格式化预测结果为可读字符串
    
    Args:
        prediction_dict: 预测结果字典
    
    Returns:
        格式化的字符串
    """
    # 错误情况
    if 'error' in prediction_dict:
        return f"""
==================
预测错误
==================
图像: {prediction_dict['image_path']}
错误: {prediction_dict['error']}
"""
    
    # 正常结果
    output = f"""
==================
预测结果
==================
图像: {prediction_dict['image_path']}
预测类别: {prediction_dict['predicted_class']}
置信度: {prediction_dict['confidence']:.2%}
"""
    
    # 添加Top-K结果
    top_k_preds = prediction_dict.get('top_k_predictions', [])
    if top_k_preds:
        output += f"\nTop-{len(top_k_preds)} 预测:\n"
        for i, (class_name, prob) in enumerate(top_k_preds, 1):
            output += f"  {i}. {class_name:15s} - {prob:6.2%}\n"
    
    return output


# 模块测试
if __name__ == "__main__":
    print("=" * 50)
    print("推理模块测试")
    print("=" * 50)
    print("\n✓ 模块加载成功!")
    print("\n可用函数:")
    print("  - load_model()")
    print("  - preprocess_input_image()")
    print("  - predict_single_image()")
    print("  - predict_batch()")
    print("  - format_predictions()")
