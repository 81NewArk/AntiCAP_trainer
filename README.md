<div align="center">

# AntiCAP_trainer

<img src=Docs/logo.jpg alt="logo" width="200" height="200">
</div>


<br>

<div align="center">

# 📄 AntiCAP_trainer 文档

<strong>AntiCAP模型训练，N卡玩具，A卡的同志请稍息。</strong>

请参阅下文了解快速安装和使用示例。有关训练、验证、预测和部署的全面指南.


</div>

<br>


## 🌍环境说明
```
# python: 3.8

# pytorch:  https://pytorch.org/

# 根据自身硬件CUDA和cuDNN版本安装torch torchvision torchaudio
```


## ⬇️ 安装说明
```
git clone https://github.com/81NewArk/AntiCAP_trainer.git

cd AntiCAP_trainer

pip install -r requirements.txt
```


## 📁 目录结构说明
```
AntiCAP_trainer/                                        # 主目录
│
├── Classification_Detection_Train/                     # 分类检测训练
│   ├── Out_Model/                                      # 训练后的模型输出
│   ├── Train_Sets/                                     # 训练数据集
│   │   ├── LABELME_DATA/                               # LabelMe格式的数据
│   │   │   ├── 0a4dfed2b7c5ead2e6e3ec00d08a5487.jpg    # 示例图像
│   │   │   ├── 0a4dfed2b7c5ead2e6e3ec00d08a5487.json   # 对应标注文件
│   │   │   ├── 00ce2c178c4a4c845992aaa33f2ccb45.jpg    # 示例图像
│   │   │   ├── 00ce2c178c4a4c845992aaa33f2ccb45.json   # 对应标注文件
│   │   ├── train/                                      # 训练集
│   │   │   ├── images/                                 # 训练图像
│   │   │   ├── labels/                                 # 训练标签
│   │   ├── val/                                        # 验证集
│   │   │   ├── images/                                 # 验证图像
│   │   │   ├── labels/                                 # 验证标签
│   ├── Classification_Detection_Train.py               # 训练脚本
│   ├── pred.py                                         # 预测脚本
│   ├── yolo11n.pt                                      # 训练好的YOLO模型
│
├── Siamese_Network_Train/                              # 孪生网络训练
│   ├── Out_Model/                                      # 训练后的模型输出
│   ├── Train_Sets/                                     # 训练数据集
│   ├── Siamese_Network_Train.py                        # 训练脚本
│   ├── pred.py                                         # 预测脚本
│   ├── vgg16-397923af.pth                              # 预训练VGG16模型
│
├── main.py  # 主入口脚本

```
<br>
<br>
<br>


# 🧰 使用方法
<br>

## 1️⃣ 目标分类检测模型训练

### 采取Labelme标注:

<br>

<img src=Docs/Text_Click_Lambel.png >


### 训练集的预处理:

<br>


```

Text_Click_CAPTCHA\Train_Sets\train\images\  # 训练前,请确保以下文件夹存在并且为空。
Text_Click_CAPTCHA\Train_Sets\train\labels\  # 训练前,请确保以下文件夹存在并且为空。
Text_Click_CAPTCHA\Train_Sets\val\images\    # 训练前,请确保以下文件夹存在并且为空。 
Text_Click_CAPTCHA\Train_Sets\val\labels\    # 训练前,请确保以下文件夹存在并且为空。

Text_Click_CAPTCHA\Train_Sets\LABELME_DATA\  # 存放 .json 和 .png|jpg 

# lamebelme标注的图片文件和对应的 .json 文件均存放于该目录
# 程序会自动划分训练集和验证集

```


<br>
<br>
<br>

## 2️⃣ 孪生网络模型训练

### 文档待编写

<br>
<br>
<br>

## 3️⃣ 自动标注

### 代码待编写

<br>
<br>
<br>


# 🐧 交流群

<div align="center">

<img src=Docs/QQ_Group.png alt="QQGroup">

</div>


<br>
<br>
<br>

# 🚬 请作者抽一包香香软软的利群
<br>

<div align="center">

<img src=Docs/Ali.png alt="Ali" width="200" height="200">
<img src=Docs/Wx.png alt="Wx" width="200" height="200">

</div>

<br>
<br>
<br>

# 🫰 致谢
<br>

[1] Ddddocr作者 网名:sml2h3


[2] 微信公众号 OneByOne 网名:十一姐


[3] 苏州大学,苏州大学文正学院 计算机科学与技术学院 张文哲教授


[4] 苏州大学,苏州大学文正学院 计算机科学与技术学院 王辉教授


[5] 苏州市职业大学,苏州大学文正学院 计算机科学与技术学院 陆公正副教授

<br>


<p>

肉麻的致谢环节终于到了。

Ddddocr的作者sml2h3无私的开源代码。项目初期，均参考于其框架。

微信公众号OneByOne作者十一姐。虽没有看付费文章，但其在公众号上分享的部分代码和文献，对项目开发提供很大的帮助。

陆公正副教授从我大专到本科，均给予我莫大的帮助。其软件工程和软件测试的课程，为软件开发测试的规范化，结构化打下坚实的基础。

王辉教授的操作系统课程，重新让我对软件开发有了全新的认知和理解。

张文哲教授，感谢您在我求学的道路上，给予我莫大的关心和帮助。

</p>

<br>
<br>
<br>

# 📚 参考文献
<br>




[1] Github. 2025.03.28 https://github.com/sml2h3


[2] Github. 2025.03.28 https://github.com/2833844911/


[3] Bilibili. 2025.03.28 https://space.bilibili.com/308704191


[4] Bilibili. 2025.03.28 https://space.bilibili.com/472467171


[5] Ultralytics. 2025.03.28 https://docs.ultralytics.com/modes/train/


[6] YRL's Blog. 2025.03.28 https://blog.2zxz.com/archives/icondetection


