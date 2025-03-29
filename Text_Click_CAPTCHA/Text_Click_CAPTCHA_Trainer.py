import os
import json
import yaml
import random
import shutil

from PIL import Image
from ultralytics import YOLO



Base_dir = os.path.abspath(os.path.dirname(__file__))                       # 项目根目录
Labelme_dir = os.path.join(Base_dir, 'Train_Sets', 'LABELME_DATA')          # Labelme数据集目录 内含json文件和图片文件
Train_Sets_dir = os.path.join(Base_dir, 'Train_Sets', 'train')              # 训练集 图片目录
Train_Labelme_dir = os.path.join(Base_dir, 'Train_Sets', 'train', 'labels') # 训练集 标签目录
Valid_Image_dir = os.path.join(Base_dir, 'Train_Sets', 'val', 'images')     # 验证集 图片目录
Valid_Labelme_dir = os.path.join(Base_dir, 'Train_Sets', 'val', 'labels')   # 验证集 标签目录
Yaml_dir = os.path.join(Base_dir, 'Train_Sets', 'train.yaml')               # yaml配置文件
Out_Model_dir = os.path.join(Base_dir, 'Out_Model')                         # 输出模型目录



def get_class_names():
    """获取所有类别，并生成类别索引映射字典"""
    class_names = set()
    for json_filename in os.listdir(Labelme_dir):
        if json_filename.endswith('.json'):
            json_path = os.path.join(Labelme_dir, json_filename)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data['shapes']:
                label = shape.get('label', '').strip()
                if label:
                    class_names.add(label)
    return {name: idx for idx, name in enumerate(sorted(class_names))}

def gen_train_yaml():
    """生成 YOLO 训练的 yaml 配置文件"""
    class_map = get_class_names()
    yaml_content = {
        'train': os.path.normpath(Train_Sets_dir).replace('\\', '/'),
        'val': os.path.normpath(Valid_Image_dir).replace('\\', '/'),
        'nc': len(class_map),
        'names': list(class_map.keys()),
    }

    # Ensure the directory for YAML file exists
    if not os.path.exists(os.path.dirname(Yaml_dir)):
        os.makedirs(os.path.dirname(Yaml_dir))

    with open(Yaml_dir, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False, allow_unicode=True)

def normalize_coordinates(width, height, xmin, ymin, xmax, ymax):
    """计算归一化坐标"""
    x_center = (xmin + xmax) / (2.0 * width)
    y_center = (ymin + ymax) / (2.0 * height)
    obj_width = (xmax - xmin) / width
    obj_height = (ymax - ymin) / height
    return x_center, y_center, obj_width, obj_height

def create_train_set():
    class_map = get_class_names()
    for json_filename in os.listdir(Labelme_dir):
        if not json_filename.endswith('.json'):
            continue

        json_path = os.path.join(Labelme_dir, json_filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_filename = os.path.basename(data['imagePath'])
        image_path = os.path.join(Labelme_dir, image_filename)

        if not os.path.exists(image_path):
            continue


        try:
            image = Image.open(image_path)
            width, height = image.size
        except (IOError, OSError) as e:
            continue


        shutil.copy(image_path, os.path.join(Train_Sets_dir, 'images', image_filename))


        txt_filename = os.path.splitext(json_filename)[0] + '.txt'
        txt_path = os.path.join(Train_Sets_dir, 'labels', txt_filename)

        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for shape in data['shapes']:
                label = shape.get('label', '').strip()
                points = shape.get('points', [])

                if not points:
                    continue

                xmin = min(p[0] for p in points)
                ymin = min(p[1] for p in points)
                xmax = max(p[0] for p in points)
                ymax = max(p[1] for p in points)

                if label not in class_map:
                    continue

                class_id = class_map[label]
                x_center, y_center, obj_width, obj_height = normalize_coordinates(
                    width, height, xmin, ymin, xmax, ymax)
                txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {obj_width:.6f} {obj_height:.6f}\n")

def create_validation_set():
    """从训练集中复制5% 最少1个 样本，作为验证集"""

    image_files = os.listdir(os.path.join(Train_Sets_dir, 'images'))
    if not image_files:
        return

    # 至少选择一个样本
    validation_count = max(1, int(len(image_files) * 0.05))  # 选择5%的图片，最少1个
    random.shuffle(image_files)
    selected_images = image_files[:validation_count]

    for image_filename in selected_images:
        image_path = os.path.join(Train_Sets_dir, 'images', image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(Train_Sets_dir, 'labels', label_filename)

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            continue

        # 创建验证集目录（如果不存在）
        os.makedirs(Valid_Image_dir, exist_ok=True)
        os.makedirs(Valid_Labelme_dir, exist_ok=True)

        # 将图像和标签文件复制到验证集目录
        shutil.copy(image_path, os.path.join(Valid_Image_dir, image_filename))
        shutil.copy(label_path, os.path.join(Valid_Labelme_dir, label_filename))



if __name__ == '__main__':
    create_train_set()
    create_validation_set()
    gen_train_yaml()




    model = YOLO('yolo11n.pt')  # 加载预训练的 YOLO 模型


    model.train(
        task='detect',            # 任务类型为检测
        data=Yaml_dir,            # 训练配置文件路径
        lr0=0.01,                 # 初始学习率
        epochs=3000,              # 训练轮数
        imgsz=640,                # 图片尺寸 640x640
        device='0',               # 使用 GPU 设备 0
        batch=16,                 # 每批次的图片数
        patience=100,             # 如果在100个epoch内没有提升，则提前停止训练
        project=Out_Model_dir,    # 输出模型的保存目录
        name='my_first_model',    # 训练模型的名称
        # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
    )