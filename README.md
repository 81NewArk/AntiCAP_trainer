<div align="center">

# AntiCAP_trainer

<img src=Docs/logo.jpg alt="logo" sizes="100px">
</div>



<br>
<br>
<br>

<div align="center">

# 📄 AntiCAP_trainer 文档

<strong>AntiCAP模型训练，N卡玩具，A卡的同志请稍息。</strong>

请参阅下文了解快速安装和使用示例。有关训练、验证、预测和部署的全面指南.


</div>

<br>
<br>
<br>

## 🌍开发环境
```
# python: 3.8

# pytorch:  https://pytorch.org/

# 根据自身硬件CUDA和cuDNN版本安装torch torchvision torchaudio
```


## ⬇️ 安装
```
git clone https://github.com/81NewArk/AntiCAP_trainer.git

cd AntiCAP_trainer

pip install -r requirements.txt
```


## 📁 目录结构及说明
```
AntiCAP_trainer/                          # 主项目文件夹
│
├── Sequ_Click_CAPTCHA/                   # "Sequence Click CAPTCHA" 相关文件夹
│   ├── pred.py                           # 用于预测
│   ├── Train_Sets/                       # 包含训练数据集的文件夹
│   │   ├── train/                        # 训练数据文件夹
│   │   │   ├── 丁                        # 验证数据集中的 "丁" 文件夹或文件
│   │   │   ├── 七                        # 验证数据集中的 "七" 文件夹或文件
│   │   ├── val/                          # 验证数据文件夹
│   │   │   ├── 丁                        # 验证数据集中的 "丁" 文件夹或文件
│   │   │   ├── 七                        # 验证数据集中的 "七" 文件夹或文件
│   │   │   └── LABELME_DATA              # Labelme注验的证集数据的 (.json文件和图片文件)
│   ├── Out_Model/                        # 模型输出目录
│   ├── Sequ_Click_CAPTCHA.py             # "Sequence Click CAPTCHA" 主要脚本
│   └── vgg16-397923af.pth                # 预训练的 VGG16 模型权重文件 （没有自行下载）
│
├── Text_Click_CAPTCHA/                   # "Text Click CAPTCHA" 相关文件夹
│   ├── Train_Sets/                       # "Text Click CAPTCHA" 训练数据集文件夹
│   │   ├── train/                        # 训练数据集文件夹
│   │   │   ├── images                    # 训练数据集图片文件夹
│   │   │   ├── labels                    # 训练数据集标签文件夹
│   │   ├── val/                          # 验证数据集文件夹
│   │   │   ├── images                    # 验证数据集图片文件夹
│   │   │   ├── labels                    # 验证数据集标签文件夹
│   ├── Out_Model/                        # 模型输出结果文件夹
│   ├── pred.py                           # 用于 "Text Click CAPTCHA" 的预测脚本
│   ├── Text_Click_CAPTCHA_Trainer.py     # 用于训练 "Text Click CAPTCHA" 模型的脚本
│   └── yolo11n.pt                        # 预训练的 YOLO 模型权重模型 (没有自行下载)
│
├── main.py                               # 项目主文件
├── README.md                             # READNE.md
└── requirements.txt                      # Python 依赖包列表文件
```
<br>
<br>
<br>


# 🧰 使用方法
<br>

## 1️⃣ 目标分类检测模型训练

### 采取Labelme标注

<br>

<img src=Docs/Text_Click_Lambel.png >


`训练集的预处理:` 训练前,请确保以下文件夹存在并且为空。

<br>

```
Text_Click_CAPTCHA\Train_Sets\train\images\
Text_Click_CAPTCHA\Train_Sets\train\labels\ 
Text_Click_CAPTCHA\Train_Sets\val\images\   
Text_Click_CAPTCHA\Train_Sets\val\labels\
```

labelme标注完成的 `.json` 和 `.jpg|.png` 文件

均存放于 `Text_Click_CAPTCHA\Train_Sets\LABELME_DATA\` 目录

程序会自动划分训练集和验证集，并转换成Yolo训练所需的 `.txt`  文件格式。

训练分类检测: `Text_Click_CAPTCHA_Trainer.py` 

模型输出目录:  `Text_Click_CAPTCHA\Out_Model\` 

<br>
<br>
<br>

## 2️⃣ 孪生网络模型训练

### 图片相似度检测

### 文档待编写

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

# 🫰 致谢

[1] Ddddocr作者 网名:sml2h3

[2] 微信公众号 OneByOne 网名:十一姐

[3] 苏州大学,苏州大学文正学院 计算机科学与技术学院 张文哲教授

[4] 苏州大学,苏州大学文正学院 计算机科学与技术学院 王辉教授

[5] 苏州市职业大学,苏州大学文正学院 计算机科学与技术学院 陆公正副教授

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

[1] Ultralytics. 2025.03.28 https://docs.ultralytics.com/modes/train/

[2] Github. 2025.03.28 https://github.com/sml2h3

[3] Github. 2025.03.28 https://github.com/2833844911/

[4] Bilibili. 2025.03.28 https://space.bilibili.com/308704191

[5] Bilibili. 2025.03.28 https://space.bilibili.com/472467171

[6] YRL's Blog. 2025.03.28 https://blog.2zxz.com/archives/icondetection

