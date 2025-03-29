<div align="center">

# AntiCAP_trainer

<img src=Docs/logo.jpg alt="logo">
</div>

<div align="center">
<br>
<br>
<br>
<b>AntiCAP模型训练，N卡玩具，A卡的同志请稍息。</b>
</div>

<br>
<br>
<br>


# 📄 AntiCAP_trainer 文档

请参阅下文了解快速安装和使用示例。有关训练、验证、预测和部署的全面指南.




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
<br>
<br>
<br>

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


## 一.目标检测

### （1） labelme 标注方法

<br>

<img src=Docs/Text_Click_Lambel.png >


### （2） 数据集预处理

<br>

训练前,请确保以下文件夹存在并且为空

`Text_Click_CAPTCHA\Train_Sets\train\images\`  
`Text_Click_CAPTCHA\Train_Sets\train\labels\`  
`Text_Click_CAPTCHA\Train_Sets\val\images\`    
`Text_Click_CAPTCHA\Train_Sets\val\labels\` 

labelme标注完成的 `.json` 和 `.jpg|.png` 文件均存放于 `Text_Click_CAPTCHA\Train_Sets\LABELME_DATA\` 目录

程序会自动划分训练集 验证集,和Yolo训练所需要的`.txt` 文件和训练格式

### （3） 训练分类检测

<br>

训练分类检测： `Text_Click_CAPTCHA_Trainer.py` 文件中有详细的训练参数注释，自行参考。

模型输出目录:  `Text_Click_CAPTCHA\Out_Model\` 


<br>
<br>
<br>

## 二. 孪生网络

<br>

todo





# 📚 参考文献

[1] Ultralytics. 2025.03.28 https://docs.ultralytics.com/modes/train/

[2] Github. 2025.03.28 https://github.com/sml2h3

[3] Github. 2025.03.28 https://github.com/2833844911/

[4] Bilibili. 2025.03.28 https://space.bilibili.com/308704191

[5] Bilibili. 2025.03.28 https://space.bilibili.com/472467171

[6] YRL's Blog. 2025.03.28 https://blog.2zxz.com/archives/icondetection

