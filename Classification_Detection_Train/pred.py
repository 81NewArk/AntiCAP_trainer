import json
from ultralytics import YOLO


# 模型路径
MODEL_PATH = r'/Classification_Detection_Train\\Out_Model\\my_first_model\\weights\\best.pt'
model = YOLO(MODEL_PATH)

# 图片路径
image_path = r'/Classification_Detection_Train\Train_Sets\train\images\0a4dfed2b7c5ead2e6e3ec00d08a5487.jpg'

# 使用YOLO模型进行推理
results = model(image_path)

# 提取检测到的框和标签
boxes = results[0].boxes
names = results[0].names

# 初始化字典来存储结果
result_dict = {}


for i in range(len(boxes)):

    label_index = int(boxes[i].cls.item())
    label = names[label_index]
    coordinates = boxes[i].xywh[0].tolist()

    # 计算框的四个角的坐标
    left_x = int(coordinates[0] - coordinates[2] / 2)
    left_y = int(coordinates[1] - coordinates[3] / 2)
    right_x = int(coordinates[0] + coordinates[2] / 2)
    right_y = int(coordinates[1] + coordinates[3] / 2)

    # 将结果存入字典
    result_dict[label] = [left_x, left_y, right_x, right_y]

# 将检测结果转换为JSON格式
json_result = json.dumps({"result": result_dict}, ensure_ascii=False)
print(json_result)

# {"result":{"像":[163,82,193,111],"章":[36,102,66,133],"每":[109,14,139,47],"近":[213,4,245,37]}}




