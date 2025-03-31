import json
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch



def pred_usePT():
    MODEL_PATH = r''
    model = YOLO(MODEL_PATH)
    image_path = r""
    results = model(image_path)

    boxes = results[0].boxes
    names = results[0].names

    result_dict = {}

    for i in range(len(boxes)):
        label_index = int(boxes[i].cls.item())
        label = names[label_index]
        coordinates = boxes[i].xywh[0].tolist()

        left_x = int(coordinates[0] - coordinates[2] / 2)
        left_y = int(coordinates[1] - coordinates[3] / 2)
        right_x = int(coordinates[0] + coordinates[2] / 2)
        right_y = int(coordinates[1] + coordinates[3] / 2)

        if label not in result_dict:
            result_dict[label] = []

        result_dict[label].append([left_x, left_y, right_x, right_y])

    json_result = json.dumps({"result": result_dict}, ensure_ascii=False)
    print("检测结果：", json_result)

    data = json.loads(json_result)
    result_dict = data["result"]

    sorted_elements = []
    for label, coordinates_list in result_dict.items():
        for coordinates in coordinates_list:
            left_x = coordinates[0]
            sorted_elements.append((left_x, label))

    sorted_elements.sort(key=lambda x: x[0])
    sorted_labels = [label for _, label in sorted_elements]
    print("使用.pt模型>>>>>>:", ''.join(sorted_labels))


def pred_useONNX():
    MODEL_PATH = r''
    image_path = r""

    # 训练时的类别名称
    label_names = [
        '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '?',
        '×', '乘', '减', '加'
    ]

    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    print(f"Model input name: {input_name}")

    image = Image.open(image_path).convert('RGB')
    image = image.resize((640, 640))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    # 获取预测输出
    output = sess.run(None, {input_name: image_array})[0]

    # 打印输出形状，查看类别和框的数量
    print(f"ONNX model output shape: {output}")


def pred_usePT_useCPU():
    MODEL_PATH = r''

    # 显式设置为 CPU 设备
    device = torch.device('cpu')

    model = YOLO(MODEL_PATH)
    model.to(device)

    image_path = r"C:\Users\75124\Desktop\555.png"


    results = model(image_path)


    boxes = results[0].boxes
    names = results[0].names

    result_dict = {}

    for i in range(len(boxes)):
        label_index = int(boxes[i].cls.item())
        label = names[label_index]
        coordinates = boxes[i].xywh[0].tolist()

        left_x = int(coordinates[0] - coordinates[2] / 2)
        left_y = int(coordinates[1] - coordinates[3] / 2)
        right_x = int(coordinates[0] + coordinates[2] / 2)
        right_y = int(coordinates[1] + coordinates[3] / 2)

        if label not in result_dict:
            result_dict[label] = []

        result_dict[label].append([left_x, left_y, right_x, right_y])


    json_result = json.dumps({"result": result_dict}, ensure_ascii=False)
    print("检测结果：", json_result)

    data = json.loads(json_result)
    result_dict = data["result"]

    sorted_elements = []
    for label, coordinates_list in result_dict.items():
        for coordinates in coordinates_list:
            left_x = coordinates[0]
            sorted_elements.append((left_x, label))

    sorted_elements.sort(key=lambda x: x[0])
    sorted_labels = [label for _, label in sorted_elements]
    print("使用.pt模型>>>>>>:", ''.join(sorted_labels))


if __name__ == '__main__':
    pred_usePT_useCPU()
