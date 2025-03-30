import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset


current_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用 {device} 进行训练")


vgg16_weights_path = os.path.join(current_dir, "vgg16-397923af.pth")
mymod = models.vgg16()

if os.path.exists(vgg16_weights_path):
    mymod.load_state_dict(torch.load(vgg16_weights_path, weights_only=True))
else:
    mymod = models.vgg16(pretrained=True)

del mymod.avgpool
del mymod.classifier


def get_dataset(path):
    """
    加载数据集，并按类别组织图片到字典中
    """
    data = {}
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        data[class_name] = [os.path.join(class_path, img) for img in os.listdir(class_path)]
    return data


def get_random_image(data, class_name, exclude=False):
    """
    获取一张来自不同类别的随机图片，exclude 为 True 时排除当前类别
    """
    class_list = list(data.keys())
    if not exclude:
        class_list.remove(class_name)  # 如果不排除当前类别，则从其他类别中选择
    random_class = class_name if exclude else random.choice(class_list)
    return random.choice(data[random_class])


def create_pairs(data):
    """
    为训练和测试生成图片对
    """
    pairs = []
    for class_name in data:
        for image_path in data[class_name]:
            positive_pair = [image_path, get_random_image(data, class_name), 0]  # 同一类别
            negative_pair = [image_path, get_random_image(data, class_name, exclude=True), 1]  # 不同类别
            pairs.extend([positive_pair, negative_pair])
    return pairs


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = mymod.features.eval().to(device)
        self.fc1 = nn.Linear(512 * 3 * 3, 512)  # 全连接层1
        self.fc2 = nn.Linear(512, 1)  # 全连接层2
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x1, x2):
        """
        前向传播，计算两张图片的相似度
        """
        x1 = torch.flatten(self.resnet(x1), 1)  # 将卷积层的输出展平
        x2 = torch.flatten(self.resnet(x2), 1)
        x = torch.abs(x1 - x2)  # 计算两张图片的绝对差异
        x = self.fc1(x)  # 通过全连接层1
        x = self.fc2(x)  # 通过全连接层2
        return self.sigmoid(x)  # 返回相似度（0到1之间）


class SiameseDataset(Dataset):
    def __init__(self, pairs, transform=None):
        """
        初始化数据集
        """
        self.pairs = pairs  # 存储图片对
        self.transform = transform  # 数据增强操作

    def __getitem__(self, idx):
        """
        获取一个数据项（图片对及其标签）
        """
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path)  # 打开图片1
        img2 = Image.open(img2_path)  # 打开图片2

        if self.transform:
            img1 = self.transform(img1)  # 对图片进行预处理
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.pairs)


def train_epoch(model, dataloader, criterion, optimizer):
    """
    训练一个epoch，计算训练损失和准确率
    """
    model.train()  # 设置模型为训练模式
    total_loss, correct, total = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Training")  # 用 tqdm 进行进度条显示

    for img1, img2, label in progress_bar:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()  # 清零梯度

        output = model(img1, img2)  # 计算网络输出
        loss = criterion(output, label.unsqueeze(1))  # 计算损失函数（这里使用二元交叉熵损失）
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累计损失
        correct += (torch.round(output) == label.unsqueeze(1)).sum().item()  # 计算正确的预测数
        total += label.size(0)  # 总样本数

        progress_bar.set_description(f"Loss: {total_loss / total:.4f} | Accuracy: {correct / total:.4f}")

    return total_loss / total  # 返回平均损失


def test_epoch(model, dataloader, criterion):
    """
    测试一个epoch，计算测试损失和准确率
    """
    model.eval()  # 设置模型为评估模式
    total_loss, correct, total = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Testing")

    with torch.no_grad():  # 测试时不需要计算梯度
        for img1, img2, label in progress_bar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            output = model(img1, img2)  # 计算网络输出
            loss = criterion(output, label.unsqueeze(1))  # 计算损失

            total_loss += loss.item()  # 累计损失
            correct += (torch.round(output) == label.unsqueeze(1)).sum().item()  # 计算正确的预测数
            total += label.size(0)  # 总样本数

            progress_bar.set_description(f"Loss: {total_loss / total:.4f} | Accuracy: {correct / total:.4f}")

    return total_loss / total  # 返回平均损失


def train():
    # 使用当前文件夹的绝对路径来加载数据集
    train_data = get_dataset(os.path.join(current_dir, 'Train_Sets', 'train'))
    val_data = get_dataset(os.path.join(current_dir, 'Train_Sets', 'val'))

    # 检查数据集加载情况
    print(f"训练数据集包含 {len(train_data)} 类，具体内容如下：")
    for class_name, images in train_data.items():
        print(f"{class_name}: {len(images)} 张图片")

    # 创建图片对
    train_pairs = create_pairs(train_data)
    print(f"训练数据对数量: {len(train_pairs)}")

    if len(train_pairs) == 0:
        print("训练数据对为空，请检查数据集路径和生成逻辑！")
        return  # 终止训练

    val_pairs = create_pairs(val_data)
    print(f"验证数据对数量: {len(val_pairs)}")

    # 定义数据增强操作
    transform = transforms.Compose([
        transforms.Resize((105, 105)),  # 调整图片大小
        transforms.RandomRotation(40),  # 随机旋转图片
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
    ])

    # 创建数据集和数据加载器
    train_dataset = SiameseDataset(train_pairs, transform=transform)
    val_dataset = SiameseDataset(val_pairs, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.BCELoss()  # 二元交叉熵损失函数

    # 模型保存路径
    model_path = os.path.join(current_dir, "Out_Model", "siamese_model.pth")

    # 加载已有模型（如果存在）
    if os.path.exists(model_path):
        print("加载已有模型进行训练")
        model.load_state_dict(torch.load(model_path))
    else:
        print("训练新模型")

    # 早停机制参数
    patience = 5  # 允许验证损失没有改进的最大epoch数
    best_val_loss = float('inf')  # 初始时，最好的验证损失为正无穷
    counter = 0  # 当前没有改进的epoch数

    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer)  # 训练一个epoch
        val_loss = test_epoch(model, val_loader, criterion)  # 测试一个epoch

        scheduler.step()

        # 如果验证损失有改善，则保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"验证损失改善，保存模型到 {model_path}")
            torch.save(model.state_dict(), model_path)
            counter = 0  # 重置计数器
        else:
            counter += 1  # 增加没有改善的epoch数

        if counter >= patience:
            print("验证损失在连续的epoch中没有改善，停止训练")
            break

if __name__ == '__main__':
    train()