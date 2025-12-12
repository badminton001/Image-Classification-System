"""
图像分类预测器类模块

封装预测逻辑，管理模型生命周期。
"""

from typing import List, Dict, Any
from .inference import load_model, predict_single_image, predict_batch


class Predictor:
    """
    图像分类预测器
    
    封装模型加载和预测功能，提供简洁的预测接口。
    
    Attributes:
        model_name: 模型名称
        weights_path: 权重文件路径
        class_names: 类别名称列表
        model: Keras模型对象
    
    使用示例:
        >>> predictor = Predictor("VGG16", "weights/vgg16.h5", class_names)
        >>> result = predictor.predict("test.jpg")
        >>> print(result['predicted_class'])
    """
    
    def __init__(self, model_name: str, weights_path: str, class_names: List[str]):
        """
        初始化预测器
        
        Args:
            model_name: 模型名称 (VGG16/ResNet50/MobileNetV2)
            weights_path: 模型权重文件路径
            class_names: 类别名称列表
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.class_names = class_names
        
        # 加载模型
        print(f"[预测器] 正在加载 {model_name}...")
        try:
            self.model = load_model(model_name, weights_path)
            print(f"[预测器] 模型加载成功!")
        except Exception as e:
            print(f"[预测器] 加载失败: {e}")
            raise
    
    def predict(self, image_path: str, top_k: int = 3) -> Dict[str, Any]:
        """
        预测单张图像
        
        Args:
            image_path: 图像文件路径
            top_k: 返回前K个预测结果，默认3
        
        Returns:
            预测结果字典
        """
        return predict_single_image(
            model=self.model,
            image_path=image_path,
            class_names=self.class_names,
            top_k=top_k
        )
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测多张图像
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            预测结果列表
        """
        return predict_batch(
            model=self.model,
            image_paths=image_paths,
            class_names=self.class_names
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = self.model.count_params()
        trainable_params = sum([
            param.numpy().size
            for param in self.model.trainable_variables
        ])
        
        return {
            'model_name': self.model_name,
            'weights_path': self.weights_path,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params)
        }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model
            print(f"[预测器] {self.model_name} 已清理")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"classes={len(self.class_names)}, "
            f"weights='{self.weights_path}')"
        )


# 模块测试
if __name__ == "__main__":
    print("=" * 50)
    print("预测器模块测试")
    print("=" * 50)
    print("\n✓ Predictor类加载成功!")
    print("\n可用方法:")
    print("  - __init__(model_name, weights_path, class_names)")
    print("  - predict(image_path, top_k=3)")
    print("  - batch_predict(image_paths)")
    print("  - get_model_info()")
