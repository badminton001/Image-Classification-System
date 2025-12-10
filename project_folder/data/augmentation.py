import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# [课件对应]: Week 10 - Part 1, Slide 17
# 老师在课件里教了 ImageDataGenerator，这里我们严格遵守课件用法。

def get_train_augmentation():
    """
    [功能]: 配置训练数据的增强器。
    [来源]: 参数设置严格参照作业要求 Task 3 [cite: 126-133]。
    """
    # 注意：这里不做 rescale (归一化)，因为 preprocessing.py 里已经做过 /255.0 了
    train_datagen = ImageDataGenerator(
        rotation_range=20,           # 旋转范围
        width_shift_range=0.2,       # 水平平移
        height_shift_range=0.2,      # 垂直平移
        horizontal_flip=True,        # 水平翻转
        zoom_range=0.2,              # 缩放范围
        brightness_range=[0.8, 1.2], # 亮度调整
        fill_mode='nearest'          # 填充模式
    )
    return train_datagen

def get_test_augmentation():
    """
    [功能]: 配置测试/验证数据的增强器。
    [注意]: 测试集永远不需要做旋转、翻转等“增强”，只需要原样输出。
    """
    # 同样不做 rescale，因为 preprocessing.py 做过了
    test_datagen = ImageDataGenerator()
    return test_datagen

def create_augmented_train_generator(X_train, y_train, batch_size=32):
    """
    [功能]: 创建训练数据的生成器 (Generator)。
    [原理]: 使用 .flow() 方法，因为我们的数据已经是加载到内存里的 Numpy 数组了。
           这符合 Week 10 课件里提到的 Generator 概念，但适配了作业要求的 Manual Loading。
    """
    train_datagen = get_train_augmentation()
    
    # .flow() 是 Keras 的标准 API，用于从内存数据生成批次
    generator = train_datagen.flow(
        X_train, 
        y_train, 
        batch_size=batch_size,
        shuffle=True  # 训练时要打乱顺序
    )
    return generator

def create_validation_generator(X_val, y_val, batch_size=32):
    """
    [功能]: 创建验证数据的生成器。
    """
    val_datagen = get_test_augmentation()
    
    generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False # 验证时不需要打乱
    )
    return generator