# 📸 图像分类增强系统 - 完整项目指南

---

## 📋 项目概述

### 项目名称
**多模型对比的图像分类增强系统**（Image Classification Enhancement System with Multi-Model Comparison）

### 项目描述
使用TensorFlow/Keras构建深度学习应用，实现：
- 使用多个预训练模型（VGG16、ResNet50、MobileNetV2）对自定义图像数据集进行分类
- 实现数据增强处理（旋转、缩放、翻转、亮度调整等）
- 自动选择性能最优的模型
- 用户可上传图像进行实时分类预测
- 可视化模型性能对比和预测结果

---

## ✅ 作业要求满足清单

### 基本要求满足
- ✅ **Python程序**: 完整的Python应用
- ✅ **TensorFlow使用**: 核心使用TensorFlow/Keras构建神经网络
- ✅ **用户交互**: 命令行菜单式交互，接受用户输入和输出预测结果
- ✅ **控制结构**: 使用while循环、for循环、if/else条件判断
- ✅ **变量与数据结构**: 使用字典、列表、numpy数组存储数据
- ✅ **函数定义**: 定义多个函数实现模块化
- ✅ **代码注释**: 完整注释说明引用来源

### 评分标准对应

| 评分项 | 权重 | 项目体现 |
|--------|------|--------|
| **Program Functionality (10分)** | 10% | 完整功能：数据加载→模型训练→预测→结果展示 |
| **Level of Effort (10分)** | 10% | 高难度：迁移学习、多模型对比、数据增强 |
| **Code Quality (5分)** | 5% | 规范：模块化、清晰注释、变量命名规范 |
| **Team Work (5分)** | 5% | 清晰分工：数据处理、模型训练、UI交互、评估可视化 |

---

## 🎯 核心功能需求

### 1. 数据处理模块
**做什么**: 
- 从指定目录加载图像数据集
- 实现图像预处理（调整大小、标准化）
- 实现多种数据增强方式
  - 随机旋转 (15°～45°)
  - 随机缩放 (80%～120%)
  - 随机水平/垂直翻转
  - 随机亮度调整
  - 随机裁剪
- 创建训练/验证/测试数据集（70%/15%/15%）

**涉及库**: 
```
TensorFlow/Keras:
  - keras.preprocessing.image.ImageDataGenerator
  - keras.utils.image_dataset_from_directory
  - tf.data API
  
其他:
  - OpenCV (cv2) - 图像读写处理
  - PIL/Pillow - 图像操作
  - NumPy - 数组操作
```

### 2. 模型架构模块
**做什么**:
- 实现3个预训练模型的加载和微调
  - **VGG16**: 经典CNN，参数量大，准确率高
  - **ResNet50**: 深层网络，残差连接，训练稳定
  - **MobileNetV2**: 轻量级，速度快，适合部署
- 冻结底层权重，只训练顶层
- 添加自定义顶层：
  - 全局平均池化
  - Dense(256, activation='relu')
  - Dropout(0.5) - 防止过拟合
  - Dense(num_classes, activation='softmax') - 输出层
- 编译模型：
  - 优化器: Adam (学习率 0.001)
  - 损失函数: categorical_crossentropy
  - 评估指标: accuracy

**涉及库**:
```
TensorFlow/Keras:
  - tf.keras.applications (VGG16, ResNet50, MobileNetV2)
  - tf.keras.layers
  - tf.keras.Sequential
  - tf.keras.Model (函数式API)
```

### 3. 训练与优化模块
**做什么**:
- 实现训练循环，使用回调函数：
  - **EarlyStopping**: 当验证精度不再提升时停止训练
  - **ReduceLROnPlateau**: 当精度不提升时降低学习率
  - **ModelCheckpoint**: 保存最优模型权重
- 设置超参数：
  - 批大小: 32
  - epochs: 50～100 (或更少，因为使用预训练模型)
  - 验证分割: 0.2
- 返回训练历史数据（loss、accuracy等）

**涉及库**:
```
TensorFlow/Keras:
  - tf.keras.callbacks.EarlyStopping
  - tf.keras.callbacks.ReduceLROnPlateau
  - tf.keras.callbacks.ModelCheckpoint
  - model.fit()
```

### 4. 模型评估模块
**做什么**:
- 在测试集上评估3个模型
- 计算以下指标：
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率 (Recall)
  - F1-Score
  - 混淆矩阵 (Confusion Matrix)
- 对比3个模型性能，自动选择最优模型
- 生成分类报告

**涉及库**:
```
scikit-learn:
  - sklearn.metrics.accuracy_score
  - sklearn.metrics.precision_score
  - sklearn.metrics.recall_score
  - sklearn.metrics.f1_score
  - sklearn.metrics.confusion_matrix
  - sklearn.metrics.classification_report
```

### 5. 预测与推理模块
**做什么**:
- 加载最优的预训练模型
- 接受用户输入的图像路径或上传图像
- 对图像进行预处理（调整大小、标准化）
- 使用模型进行预测
- 输出：
  - 预测类别
  - 置信度（概率）
  - Top-3 预测结果

**涉及库**:
```
TensorFlow/Keras:
  - model.predict()
  - tf.keras.preprocessing.image.load_img
  - tf.keras.preprocessing.image.img_to_array
```

### 6. 可视化模块
**做什么**:
- 绘制3个模型的性能对比（准确率、精确率等）
- 绘制混淆矩阵热图
- 绘制训练历史曲线（loss和accuracy）
- 显示预测结果的概率分布柱状图
- 显示原图像和预测结果

**涉及库**:
```
Matplotlib:
  - plt.bar() - 柱状图
  - plt.plot() - 曲线图
  - plt.imshow() - 显示图像
  
Seaborn (可选):
  - sns.heatmap() - 混淆矩阵热图
```

### 7. 用户交互模块
**做什么**:
- 实现命令行菜单系统
- 菜单选项：
  1. 选择数据集并预处理
  2. 选择要训练的模型（VGG16/ResNet50/MobileNetV2/全部）
  3. 训练选定的模型
  4. 评估模型并对比性能
  5. 使用最优模型进行单个预测
  6. 批量预测多个图像
  7. 可视化结果
  8. 退出程序
- 输入验证和错误处理
- 显示进度条和训练日志

**涉及库**:
```
标准库:
  - input() - 用户输入
  - print() - 输出显示
  - os - 文件操作
  - time - 计时
  
第三方:
  - tqdm - 进度条
```

---

## 📁 项目文件结构（必须）

```
GroupID_FinalProject.zip
│
├── GroupID_Cover.pdf                    ← 必需
│   ├ 项目标题: 多模型对比的图像分类增强系统
│   ├ 组号: GroupID
│   ├ 团队成员: 5-6人（名字+学号）
│   ├ 声明: 原创声明
│   └ 评分表: 附上官方评分标准
│
├── README.txt                           ← 必需
│   ├ 项目说明
│   ├ 使用说明（如何运行）
│   ├ 文件结构说明
│   ├ 数据集准备说明
│   ├ 依赖包列表
│   └ 预期结果
│
├── main.py                              ← 必需（一键运行入口）
│   └ 包含完整的控制流和菜单
│
└── project_folder/
    ├── main.py                          # 主程序（与上面相同或更详细版本）
    │
    ├── config.py                        # 配置文件
    │   ├ 数据集路径
    │   ├ 模型参数
    │   ├ 训练参数
    │   └ 其他常量
    │
    ├── data/
    │   ├── data_loader.py              # 数据加载函数
    │   │   ├ load_dataset_from_directory()
    │   │   ├ load_and_preprocess_image()
    │   │   └ create_data_generators()
    │   │
    │   ├── preprocessing.py            # 数据预处理
    │   │   ├ resize_image()
    │   │   ├ normalize_image()
    │   │   ├ split_dataset()
    │   │   └ prepare_dataset()
    │   │
    │   └── augmentation.py             # 数据增强
    │       ├ ImageDataGenerator配置
    │       ├ custom_augmentation()
    │       └ create_augmented_batches()
    │
    ├── models/
    │   ├── model_architecture.py       # 模型定义
    │   │   ├ build_vgg16_model()
    │   │   ├ build_resnet50_model()
    │   │   ├ build_mobilenetv2_model()
    │   │   └ compile_model()
    │   │
    │   ├── train.py                    # 训练逻辑
    │   │   ├ train_single_model()
    │   │   ├ train_all_models()
    │   │   └ get_callbacks()
    │   │
    │   └── evaluate.py                 # 评估逻辑
    │       ├ evaluate_model()
    │       ├ compare_models()
    │       ├ calculate_metrics()
    │       ├ get_confusion_matrix()
    │       └ get_best_model()
    │
    ├── predict/
    │   ├── inference.py               # 推理
    │   │   ├ load_model()
    │   │   ├ predict_single_image()
    │   │   ├ predict_batch()
    │   │   ├ preprocess_input_image()
    │   │   └ format_predictions()
    │   │
    │   └── predictor.py               # 预测器类
    │       └ Predictor class
    │
    ├── utils/
    │   ├── visualization.py           # 可视化
    │   │   ├ plot_model_comparison()
    │   │   ├ plot_confusion_matrix()
    │   │   ├ plot_training_history()
    │   │   ├ plot_prediction_distribution()
    │   │   └ display_image_with_prediction()
    │   │
    │   ├── metrics.py                 # 性能指标计算
    │   │   ├ calculate_accuracy()
    │   │   ├ calculate_precision()
    │   │   ├ calculate_recall()
    │   │   ├ calculate_f1_score()
    │   │   └ get_classification_report()
    │   │
    │   ├── helpers.py                 # 辅助函数
    │   │   ├ print_menu()
    │   │   ├ get_user_choice()
    │   │   ├ validate_input()
    │   │   ├ format_output()
    │   │   ├ save_results()
    │   │   └ load_results()
    │   │
    │   └── logger.py                  # 日志记录
    │       └ setup_logger()
    │
    ├── ui/
    │   └── interface.py               # 用户交互
    │       ├ UserInterface class
    │       ├ main_menu()
    │       ├ dataset_menu()
    │       ├ training_menu()
    │       ├ prediction_menu()
    │       └ visualization_menu()
    │
    ├── weights/
    │   ├── vgg16_best_model.h5        # 保存的最优模型
    │   ├── resnet50_best_model.h5
    │   └── mobilenetv2_best_model.h5
    │
    ├── results/
    │   ├── metrics.txt                # 保存的性能指标
    │   ├── confusion_matrix.png       # 混淆矩阵图像
    │   ├── model_comparison.png       # 模型性能对比
    │   ├── training_history.png       # 训练曲线
    │   └── predictions.json           # 预测结果
    │
    ├── data_samples/                   # 示例数据集
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class_2/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── class_3/
    │
    ├── requirements.txt               # 必需依赖包
    │
    └── .gitignore                     # 忽略文件（大型模型文件等）
        ├ *.h5
        ├ __pycache__/
        ├ .DS_Store
        └ *.pyc
```

---

## 📦 依赖包清单 (requirements.txt)

```
tensorflow>=2.10.0
keras>=2.10.0
opencv-python>=4.5.0
pillow>=9.0.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
tqdm>=4.60.0
```

### 安装命令
```bash
pip install -r requirements.txt
```

---

## 🚀 运行方式

### 最简单的运行方式 (for 评阅老师)
```bash
# 在GroupID_FinalProject目录下
python main.py
```

### 完整运行流程
1. **准备数据集**
   - 创建 `project_folder/data_samples/` 目录
   - 在其中创建子目录（每个类别一个目录）
   - 放入图像文件（JPG、PNG格式）

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行程序**
   ```bash
   cd project_folder
   python main.py
   ```

4. **按菜单操作**
   ```
   ===== 图像分类增强系统 =====
   1. 加载并预处理数据集
   2. 训练单个模型
   3. 训练全部模型
   4. 评估并对比模型
   5. 预测单个图像
   6. 批量预测
   7. 可视化结果
   8. 退出
   
   请选择 (1-8): 
   ```

---

## 🔑 关键实现细节

### 1. 数据增强配置示例

```python
# 使用ImageDataGenerator进行数据增强
train_datagen = ImageDataGenerator(
    rotation_range=20,           # 随机旋转 20度
    width_shift_range=0.2,       # 水平平移 20%
    height_shift_range=0.2,      # 垂直平移 20%
    horizontal_flip=True,        # 水平翻转
    zoom_range=0.2,              # 缩放 20%
    brightness_range=[0.8, 1.2], # 亮度调整
    fill_mode='nearest'          # 补充方式
)
```

### 2. 预训练模型加载示例

```python
# 加载预训练权重，冻结底层，自定义顶层
base_model = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,           # 不包含顶层
    weights='imagenet'           # 使用ImageNet预训练权重
)

# 冻结底层权重
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义顶层
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### 3. 回调函数配置示例

```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,               # 5个epoch没有改善则停止
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,               # 学习率降低到50%
        patience=3,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True       # 只保存最优模型
    )
]
```

### 4. 性能指标计算示例

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 获取预测结果
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 计算指标
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

# 混淆矩阵和分类报告
cm = confusion_matrix(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes)
```

---

## 📊 README.txt 示例内容

```
========================================
   多模型对比的图像分类增强系统
   Image Classification Enhancement System
========================================

【项目说明】
本项目使用TensorFlow/Keras深度学习框架，实现：
1. 加载和预处理图像数据集
2. 使用三个预训练模型进行迁移学习
3. 对比模型性能并自动选择最优模型
4. 支持用户上传图像进行实时分类预测
5. 可视化训练过程和预测结果

【使用说明】
1. 准备数据集：
   - 在 data_samples/ 目录下创建子目录（一个类别一个目录）
   - 在每个子目录中放入该类别的图像文件

2. 安装依赖：
   pip install -r requirements.txt

3. 运行程序：
   python main.py

4. 按菜单选择操作

【文件结构】
- main.py: 主程序入口
- data/: 数据加载和预处理模块
- models/: 模型定义和训练模块
- predict/: 预测推理模块
- utils/: 可视化和工具函数
- ui/: 用户交互界面
- weights/: 保存的模型权重

【数据集格式】
data_samples/
├── class_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── class_2/
│   ├── image_1.jpg
│   └── ...
└── class_3/

【模型说明】
- VGG16: 经典CNN，参数量大，准确率高
- ResNet50: 深层网络，训练稳定
- MobileNetV2: 轻量级，速度快

【预期结果】
- 三个模型的训练曲线和性能对比
- 混淆矩阵和分类报告
- 用户上传图像的预测结果和置信度

【技术栈】
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-learn
- Matplotlib
- NumPy

【注意事项】
- 确保数据集中每个类别至少有5-10张图像
- 建议使用JPG或PNG格式的图像
- 第一次运行会下载预训练权重，需要网络连接
- 模型权重会保存在 weights/ 目录
```

---

## ✅ 代码质量检查清单

### 必须包含的代码规范

| 检查项 | 说明 | 示例 |
|--------|------|------|
| **注释完整** | 文件头、函数说明、关键代码行注释 | `# 加载预训练模型权重` |
| **引用注明** | 使用外部代码必须注明来源URL | `# 参考: https://tensorflow.org/...` |
| **函数名规范** | 小写+下划线，动词开头 | `load_dataset()`, `train_model()` |
| **变量名规范** | 清晰易懂，避免单字母 | `image_data` 而非 `x` |
| **错误处理** | try-except捕获异常 | 文件不存在、路径错误等 |
| **进度反馈** | 长时间操作显示进度 | 使用tqdm进度条 |
| **配置文件** | 参数不硬编码，使用config.py | 模型名称、路径、超参数 |
| **日志记录** | 重要操作记录日志 | 训练开始、模型保存等 |

---

## 🎯 获得高分的关键策略

### Program Functionality (10分) - 完整性
✅ 一键运行 (main.py 正常运行)
✅ 完整的数据→训练→评估→预测流程
✅ 用户菜单完善，多个选项
✅ 结果保存和导出

### Level of Effort (10分) - 难度与创新
✅ 使用TensorFlow高级特性（迁移学习、多模型对比）
✅ 实现数据增强处理
✅ 自动模型选择逻辑
✅ 性能对比和可视化
✅ 模型保存和加载

### Code Quality (5分) - 代码质量
✅ 模块化设计（分离concerns）
✅ 完整的代码注释和文档
✅ 规范的变量和函数命名
✅ 错误处理和输入验证
✅ 配置文件管理参数

### Team Work (5分) - 团队协作
✅ 清晰的模块分工
✅ 各模块独立且互相配合
✅ 统一的代码风格
✅ 集成测试确保功能完整

---

## 📝 最后检查清单

提交前必须确认：

- [ ] 文件结构完整（Cover.pdf、README.txt、main.py等）
- [ ] 代码可直接运行（python main.py）
- [ ] requirements.txt列出所有依赖
- [ ] 包含示例数据集 (data_samples/)
- [ ] 所有引用代码标注来源URL
- [ ] 所有函数都有文档字符串
- [ ] 关键代码行都有注释
- [ ] 错误处理完善（try-except）
- [ ] 用户输入验证（防止崩溃）
- [ ] 结果可视化和保存
- [ ] 模型权重可自动下载和保存
- [ ] README说明清晰（如何运行、数据格式等）
- [ ] Cover.pdf包含评分表
- [ ] 团队成员分工明确
