import os
import shutil
import numpy as np
from PIL import Image
import unittest

# 引用你的模块
from data.data_loader import load_dataset_from_directory, validate_dataset
from data.preprocessing import prepare_dataset
from data.augmentation import create_augmented_train_generator

class TestDataModule(unittest.TestCase):
    """
    [功能]: 自动测试数据处理模块的所有功能。
    [原理]: 创建一个临时的假数据集 -> 运行代码 -> 检查结果 -> 删除假数据。
    """

    def setUp(self):
        """测试开始前的准备工作：造假数据"""
        self.test_dir = "temp_test_data"
        self.classes = ["cat", "dog"]
        
        # 1. 创建临时文件夹
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # 2. 在每个类别里造 10 张假图片
        for class_name in self.classes:
            class_path = os.path.join(self.test_dir, class_name)
            os.makedirs(class_path)
            for i in range(10):
                # 创建一个 100x100 的随机RGB图片
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(class_path, f"img_{i}.jpg"))
                
        print(f"\n[Setup] Created temporary dataset at {self.test_dir}")

    def tearDown(self):
        """测试结束后的清理工作：删除假数据"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("[Teardown] Removed temporary dataset")

    def test_1_data_loader(self):
        """测试数据加载器"""
        print("--- Testing Data Loader ---")
        paths, labels, class_names = load_dataset_from_directory(self.test_dir)
        
        # 断言检查：应该有 20 张图 (10 cat + 10 dog)
        self.assertEqual(len(paths), 20)
        # 断言检查：应该有 2 个类别
        self.assertEqual(len(class_names), 2)
        print("Data Loader Test Passed!")

    def test_2_preprocessing(self):
        """测试预处理流程 (切分 + 归一化 + 缩放)"""
        print("--- Testing Preprocessing ---")
        # 运行你的核心函数
        (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = prepare_dataset(
            self.test_dir, target_size=(50, 50), test_size=0.2, val_size=0.2
        )
        
        # 检查是否成功返回
        self.assertIsNotNone(X_train)
        
        # 检查形状：X 应该是 (数量, 50, 50, 3)
        self.assertEqual(X_train.shape[1:], (50, 50, 3))
        
        # 检查归一化：最大值不应超过 1.0
        self.assertTrue(X_train.max() <= 1.0)
        
        print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        print("Preprocessing Test Passed!")

    def test_3_augmentation(self):
        """测试数据增强生成器"""
        print("--- Testing Augmentation ---")
        # 造一些假数据给生成器
        X_dummy = np.random.rand(10, 50, 50, 3).astype('float32')
        y_dummy = np.random.randint(0, 2, 10)
        
        # 调用你的增强函数
        generator = create_augmented_train_generator(X_dummy, y_dummy, batch_size=4)
        
        # 尝试从生成器里拿一批数据
        batch_x, batch_y = next(generator)
        
        # 检查批次大小
        self.assertEqual(batch_x.shape[0], 4)
        print("Augmentation Generator Test Passed!")

if __name__ == '__main__':
    unittest.main()