# 手势识别项目学习指南（初学者版）

> 这份文档面向**第一次接触计算机视觉 / 手势识别**的同学，按照"一个一个讲"的节奏，
> 把 `main.py` 里的每一个知识点拆开讲清楚。
>
- 阅读建议：**跟着章节顺序读，不要跳**，每一章都以前一章为基础
- 遇到代码先不要慌，文档里都有"这段代码在干什么"的逐行解释
- 每章末尾有小练习，动手改一改代码，比光看有效十倍

---

## 目录

- [第一章 项目是什么？能做什么？](#第一章-项目是什么能做什么)
- [第二章 三个核心库各是干什么的](#第二章-三个核心库各是干什么的)
- [第三章 环境准备与项目结构](#第三章-环境准备与项目结构)
- [第四章 必须先懂的图像基础](#第四章-必须先懂的图像基础)
- [第五章 MediaPipe 21 个手部关键点](#第五章-mediapipe-21-个手部关键点)
- [第六章 主程序逐段精读（main.py）](#第六章-主程序逐段精读-mainpy)
- [第七章 0~5 数字手势的判定原理](#第七章-05-数字手势的判定原理)
- [第八章 自测脚本怎么读](#第八章-自测脚本怎么读selftestpy)
- [第九章 动手练习（难度递增）](#第九章-动手练习难度递增)
- [第十章 常见问题 FAQ](#第十章-常见问题-faq)
- [第十一章 进阶学习路线](#第十一章-进阶学习路线)
- [附录 A 21 个关键点完整索引表](#附录-a-21-个关键点完整索引表)
- [附录 B 常用 OpenCV API 速查](#附录-b-常用-opencv-api-速查)
- [附录 C 术语表](#附录-c-术语表)

---

## 第一章 项目是什么？能做什么？

### 1.1 一句话介绍

用电脑摄像头"看"你的手，在屏幕上**画出手部骨架**，并识别你**伸出了几根手指**（0~5）。

### 1.2 最终效果

- 画面里出现一只手：手上画出 21 个点和连线（骨架）；
- 握拳：显示"数字 0"；伸 1 根手指：显示"数字 1"……张开手掌：显示"数字 5"；
- 最多同时识别 **2 只手**；
- 按键盘 **`q`** 键退出。

### 1.3 程序每一秒都在做什么

摄像头每秒大约吐出 30 张图片（叫"帧"）。程序对**每一帧**都重复下面这条流水线：

```mermaid
flowchart LR
    A[摄像头取一帧画面] --> B[水平镜像变成自拍视角]
    B --> C[交给 MediaPipe 检测手部]
    C --> D[得到 21 个关键点坐标]
    D --> E[画出骨架连线和关键点]
    E --> F[判断伸出几根手指]
    F --> G[在画面上写中文结果]
    G --> H[显示窗口]
    H -->|下一帧| A
```

**理解这张图，你就理解了整个项目。** 后面的所有代码都是在实现这条流水线。

### 1.4 先把项目跑起来

在讲原理之前，先亲手跑一遍，有个直观感受：

1. 双击项目里的 **`run.bat`**（首次会自动装依赖，需要几分钟）；
2. 看到摄像头窗口后，把手举到镜头前；
3. 握拳、伸手指，观察屏幕上的数字变化；
4. 按 `q` 退出。

---

## 第二章 三个核心库各是干什么的

整个项目用到四个第三方库，各司其职，**记住它们的分工**是理解代码的第一步：

| 库名 | 安装名 | 在项目里负责什么 | 通俗比喻 |
| --- | --- | --- | --- |
| **OpenCV** | `opencv-python` | 打开摄像头、读取画面、画线画圆、显示窗口 | 眼睛和画笔 |
| **MediaPipe** | `mediapipe` | 识别手部 21 个关键点（AI 模型） | 一个训练好的"手部专家" |
| **Pillow** | `Pillow` | 在画面上写中文（OpenCV 自带的写字函数不支持中文） | 会写中文的便签笔 |
| **NumPy** | `numpy` | 把图片当成数字矩阵处理（其它库会自动安装它） | 数字表格 |

### 2.1 OpenCV（cv2）

- 计算机视觉领域最常用的开源库；
- 代码里用 `import cv2` 导入（注意：**导入名是 `cv2`，不是 `opencv`**）；
- 本项目用到的能力：
  - `cv2.VideoCapture()`：打开摄像头；
  - `cv2.cvtColor()`：转换颜色格式（BGR → RGB）；
  - `cv2.flip()`：画面翻转（镜像）；
  - `cv2.line()` / `cv2.circle()`：画线段和圆点；
  - `cv2.imshow()` / `cv2.waitKey()`：显示窗口、等待按键。

### 2.2 MediaPipe

- Google 开源的机器学习工具集，里面有很多"现成的、训练好的"人体感知模型；
- 本项目使用其中的 **HandLandmarker（手部关键点检测器）**；
- 它的输出是：**每只手 21 个关键点**的坐标（见第五章）；
- 它需要搭配一个模型文件 `models/hand_landmarker.task`（约 7.8MB），可以理解成"专家的大脑"。

> ⚠️ **重要提醒（容易踩坑）**：
> MediaPipe 1.x 版本已经移除了旧教程里常见的 `mp.solutions.hands` 接口。
> 本项目使用官方现在推荐的新接口 **Tasks API**（`HandLandmarker`）。
> 新、旧接口都能输出 21 个关键点，但写法不同，本项目代码是按新接口写的。

### 2.3 Pillow（PIL）

- 经典的 Python 图像处理库；
- 本项目只用它做一件事：**在画面上写中文**。
- 为什么不用 OpenCV 的 `cv2.putText()`？因为它不支持中文汉字，写出来会变成"？？？"。

### 2.4 小练习

- 打开 `main.py`，数一数一共出现了哪几个 `import`；
- 思考：如果不装 Pillow，程序还能显示中文吗？（答案在第七章末尾 FAQ）

---

## 第三章 环境准备与项目结构

### 3.1 安装命令（要记住的三条 pip）

```bash
pip install opencv-python
pip install mediapipe
pip install pillow
```

或一次性安装：

```bash
pip install -r requirements.txt
```

国内网络慢时可以用国内镜像加速：

```bash
pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

### 3.2 为什么要用虚拟环境（venv）

初学者常见困惑：`pip` 安装的库放在哪？

- 不做任何设置时，库会装到你的**全局 Python 环境**里；
- 不同项目可能需要不同版本的库，全装在一个地方容易"打架"（比如 A 项目要 numpy 1.x，B 项目要 numpy 2.x）；
- **虚拟环境（venv）** 相当于给项目单独准备一个"独立工具箱"，里面的库只服务这个项目，不影响别的项目。

本项目已经自动建好了一个名为 `.venv` 的虚拟环境。相关命令了解即可：

```bash
# 1. 创建虚拟环境（只需一次）
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # macOS / Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 退出虚拟环境
deactivate
```

> 不想每次激活也可以直接用虚拟环境里的 Python：
> `.venv\Scripts\python.exe main.py`
> 这就是 `run.bat` 一键启动的原理。

### 3.3 项目文件结构

```
python手势识别/
├── run.bat                  # Windows 一键运行（双击即可）
├── self_test.bat            # Windows 一键自测
├── main.py                  # ★ 主程序，所有核心代码都在这
├── self_test.py             # 自测脚本（检查环境和逻辑对不对）
├── requirements.txt         # 依赖清单
├── models/
│   └── hand_landmarker.task # MediaPipe 手部模型（专家的大脑）
├── assets/
│   ├── test_hands.jpg       # 自测用的静态手部图片
│   └── demo_result.jpg      # 识别效果图
├── README.md                # 项目说明（怎么安装、怎么运行）
├── LEARNING_GUIDE.md        # ← 你正在看的这份学习文档
└── .gitignore               # 告诉 Git 哪些文件不需要上传
```

**学习顺序建议**：先读 `main.py` 的代码（本指南第六章带读），遇到 `self_test.py` 里的概念再回看第八章。

---

## 第四章 必须先懂的图像基础

这一章讲 4 个"不懂就看不懂后面代码"的基础概念。

### 4.1 像素：画面就是数字表格

一张图片由很多个小格子组成，每个小格子叫一个**像素（pixel）**。
比如 640×480 的画面，就是 640 列 × 480 行，一共 307200 个像素。

每个像素用 **3 个数字**表示颜色（蓝、绿、红各一个分量，范围 0~255）：

| 像素颜色 | B（蓝） | G（绿） | R（红） |
| --- | --- | --- | --- |
| 黑色 | 0 | 0 | 0 |
| 白色 | 255 | 255 | 255 |
| 纯红 | 0 | 0 | 255 |
| 纯绿 | 0 | 255 | 0 |
| 纯蓝 | 255 | 0 | 0 |
| 黄色 | 0 | 255 | 255 |

### 4.2 图像坐标系：y 轴朝下！

这是**最重要、最容易搞反**的一个概念：

- 数学里坐标 y 轴**向上**，而图像里 y 轴**向下**；
- 左上角是原点 `(0, 0)`；
- 越往右 x 越大，越往下 y 越大。

```
(0,0) ──── x 增大 ────→
  │
  │      手指向上伸直时
  │      指尖在关节"上方"
  y       即 y 更小
  增
  大
  ↓
```

**手指计数的核心判断就靠这个**：手指向上伸直时，指尖比中间关节更靠上，
所以"指尖的 y 小于关节的 y"。第七章会细讲。

### 4.3 BGR 与 RGB 的区别

- 人习惯说"红、绿、蓝"（**RGB**）；
- OpenCV 存储像素时顺序是**蓝、绿、红**（**BGR**）；
- MediaPipe 模型要求输入 **RGB**。

所以代码里要先做一次转换：

```python
# OpenCV 读到的是 BGR，MediaPipe 要 RGB，这里做转换
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

> 记住：**OpenCV 用 BGR，MediaPipe 用 RGB**，中间必须 `cvtColor`。

### 4.4 归一化坐标：0~1 而不是像素

MediaPipe 输出的关键点坐标不是"第几个像素"，而是 **0~1 之间的小数**（归一化坐标）：

- `x = 0.5` 表示在画面水平方向的正中间；
- `y = 0.0` 表示画面最顶端，`y = 1.0` 表示最底端。

好处：**和分辨率无关**。无论摄像头是 640×480 还是 1280×720，
同一只手的归一化坐标都是一样的。

换算成真正的像素坐标很简单：

```python
h, w = image.shape[:2]                  # 得到画面高度、宽度（像素）
pixel_x = int(landmark.x * w)           # 归一化 x × 画面宽度
pixel_y = int(landmark.y * h)           # 归一化 y × 画面高度
```

### 4.5 为什么要把画面镜像（翻转）

摄像头默认画面里，你举起右手，屏幕中的手在左边（像照镜子），看着别扭。
代码用 `cv2.flip(frame, 1)` 做**水平镜像**，让画面和照镜子一样：
你举右手，屏幕里的手也在右边，体验更自然。

```python
# 1 表示水平翻转（左右镜像）；0 表示上下翻转；-1 表示两个方向都翻转
frame = cv2.flip(frame, 1)
```

### 4.6 小练习

1. 代码里 `color=(0, 255, 0)` 画出来的是什么颜色？（BGR → 绿色）
2. 一个点归一化坐标 `x=0.2, y=0.3`，在 1000×1000 画面上的像素坐标是多少？（200, 300）
3. 为什么把画面送进 MediaPipe 之前要先 `cvtColor`？

---

## 第五章 MediaPipe 21 个手部关键点

### 5.1 一只手 = 21 个点

MediaPipe 会在你的手上找到 21 个固定位置的点：
1 个手腕点 + 5 根手指各 4 个点。编号 0~20：

```
                          中指
                    9 ──10 ──11 ──12   (MCP-PIP-DIP-TIP)
      食指                     无名指
  5 ──6 ──7 ──8          13 ──14 ──15 ──16
       \                    /
0 手腕 ── ... 手掌骨架 ...  ── 17 ──18 ──19 ──20  小指
 │
 1 ── 2 ── 3 ── 4   拇指（CMC-MCP-IP-TIP）
```

### 5.2 每根手指的四段命名

每根手指从手掌往外数，4 个点的名称分别是：

| 缩写 | 英文 | 中文 | 位置 |
| --- | --- | --- | --- |
| CMC | Carpometacarpal joint | 腕掌关节 | 仅拇指有，拇指根部 |
| MCP | Metacarpophalangeal joint | 掌指关节 | 手指和手掌连接处（"指根"） |
| PIP | Proximal Interphalangeal joint | 近端指间关节 | 中间的关节 |
| DIP | Distal Interphalangeal joint | 远端指间关节 | 靠近指尖的关节 |
| TIP | Tip | 指尖 | 手指最顶端 |

> 记忆口诀：**"根、中、尖、顶"** —— 从手掌往外，MCP（根）、PIP（中）、DIP（尖）、TIP（顶）。

### 5.3 21 个点编号对照表

| 编号 | 名称 | 说明 |
| --- | --- | --- |
| 0 | WRIST | 手腕 |
| 1 | THUMB_CMC | 拇指根（腕掌关节） |
| 2 | THUMB_MCP | 拇指掌指关节 |
| 3 | THUMB_IP | 拇指指间关节 |
| **4** | **THUMB_TIP** | **拇指尖** |
| 5 | INDEX_MCP | 食指根 |
| 6 | INDEX_PIP | 食指近端关节 |
| 7 | INDEX_DIP | 食指远端关节 |
| **8** | **INDEX_TIP** | **食指尖** |
| 9 | MIDDLE_MCP | 中指根 |
| 10 | MIDDLE_PIP | 中指近端关节 |
| 11 | MIDDLE_DIP | 中指远端关节 |
| **12** | **MIDDLE_TIP** | **中指尖** |
| 13 | RING_MCP | 无名指根 |
| 14 | RING_PIP | 无名指近端关节 |
| 15 | RING_DIP | 无名指远端关节 |
| **16** | **RING_TIP** | **无名指尖** |
| 17 | PINKY_MCP | 小指根 |
| 18 | PINKY_PIP | 小指近端关节 |
| 19 | PINKY_DIP | 小指远端关节 |
| **20** | **PINKY_TIP** | **小指尖** |

判断手指是否伸直时，最常用的是 **TIP（指尖）和 PIP（近端关节）**，表中加粗的是指尖。

### 5.4 骨架连接关系

点有了，哪些点之间画连线？项目里用 `HAND_CONNECTIONS` 列表定义：

```python
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),  # 手掌部分
    (1, 2), (2, 3), (3, 4),          # 拇指
    (5, 6), (6, 7), (7, 8),          # 食指
    (9, 10), (10, 11), (11, 12),     # 中指
    (13, 14), (14, 15), (15, 16),    # 无名指
    (17, 18), (18, 19), (19, 20),    # 小指
]
```

每个元组 `(起点编号, 终点编号)` 表示这两个点之间画一条线。共 21 条连线。

### 5.5 检测器的三种运行模式

创建 HandLandmarker 时要选一个**运行模式**（`RunningMode`）：

| 模式 | 用途 | 对应方法 |
| --- | --- | --- |
| `IMAGE` | 单张静态图片（如照片） | `detect(图片)` |
| `VIDEO` | 摄像头视频流（**本项目使用**） | `detect_for_video(图片, 时间戳)` |
| `LIVE_STREAM` | 直播流，异步回调 | `detect_async(图片, 时间戳)` |

视频模式必须传入**每一帧单调递增的时间戳**（毫秒），否则会报错。
代码里用帧编号 × 33 来生成（按 30fps 估算，每帧约 33ms）：

```python
timestamp_ms = frame_index * 33
result = landmarker.detect_for_video(mp_image, timestamp_ms)
```

### 5.6 检测结果长什么样

检测结果（`result`）里最重要的是两个列表：

- `result.hand_landmarks`：关键点列表。检测到 2 只手时它长度为 2；
  每只手是一个含 21 个点的列表，每个点有 `.x`、`.y`（归一化坐标）；
- `result.handedness`：左右手标签列表，和 `hand_landmarks` 一一对应；
  每个元素 `[0].category_name` 是字符串 `"Left"` 或 `"Right"`。

访问方式：

```python
for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
    label = handedness[0].category_name   # "Left" 或 "Right"
    tip_of_index = landmarks[8]           # 食指尖（第 8 号点）
    print(tip_of_index.x, tip_of_index.y) # 打印归一化坐标
```

### 5.7 小练习

1. 说出小指尖、拇指尖、中指根的编号。（20、4、9）
2. 检测 3 只手时，项目当前能识别到吗？（提示：`num_hands=2`，要识别 3 只需要改参数）
3. 为什么视频模式要传时间戳？

---

## 第六章 主程序逐段精读（main.py）

这一章按代码顺序，把 `main.py` 拆成 10 个小段逐段讲解。

### 6.1 导入部分

```python
import argparse
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont
```

逐个说明：

- `argparse`：解析命令行参数（如 `--camera 1`）；
- `os`、`sys`：系统操作、程序退出；
- `urllib.request`：下载模型文件；
- `pathlib.Path`：用面向对象的方式处理文件路径（比手写字符串拼接更不容易出错）；
- `GLOG_minloglevel=2`：**必须在 import mediapipe 之前设置**，作用是屏蔽 MediaPipe 启动时刷的一大堆日志（只保留报错）；
- 最后 5 行是第三方库导入，见第二章。

### 6.2 常量定义

```python
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
```

`__file__` 表示当前这个 `main.py` 文件；`resolve().parent` 取它所在的目录。
这样无论你从哪个目录启动程序，都能正确找到项目里的 `models/` 文件夹。

随后是 21 个关键点的编号常量和骨架连接表，见第五章。

### 6.3 模型下载：`ensure_model()`

```python
def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024:
        return True                       # 模型已存在且大于 1MB，直接使用
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for url in MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, MODEL_PATH, reporthook=...)
            if MODEL_PATH.stat().st_size > 1024 * 1024:
                return True
        except Exception as exc:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()       # 删除下载了一半的坏文件
    return False
```

做的事：

1. 模型文件存在且大于 1MB → 直接返回 `True`；
2. 不存在 → 自动创建 `models/` 目录并从官方地址下载；
3. 下载过程中显示进度百分比；
4. 下载失败 → 删除不完整的临时文件，给出手动下载提示。

> 这是"容错处理"的第一个体现：**即使文件缺失，程序也能自动恢复，而不是直接崩溃**。

### 6.4 创建检测器：`create_landmarker()`

```python
model_buffer = MODEL_PATH.read_bytes()    # 先把模型读成字节流
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=model_buffer),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)
```

参数逐个解释：

| 参数 | 含义 |
| --- | --- |
| `model_asset_buffer` | 模型的字节内容。**为什么不用文件路径？** 因为 MediaPipe 底层 C++ 在 Windows 上不支持中文路径，而本项目路径含中文，读成字节流传入可以彻底规避 |
| `running_mode` | 视频流模式 |
| `num_hands` | 最多检测几只手，这里设 2 |
| `min_hand_detection_confidence` | 手部检测置信度阈值（0~1），越高越严格 |
| `min_hand_presence_confidence` | 手部存在置信度阈值 |
| `min_tracking_confidence` | 跨帧追踪的置信度阈值 |

> 识别时灵时不灵？可以试试把 0.5 调低（如 0.3，更灵敏但可能误识别）或调高（如 0.7，更严格）。

### 6.5 手指计数函数：`count_fingers()`（核心中的核心）

这是整个项目"灵魂"所在，完整代码和第七章配合阅读，这里先看结构：

```python
def count_fingers(landmarks, handedness_label):
    fingers = []
    # 第一步：判断四根手指（食、中、无名、小）
    for tip_id, pip_id in ((8,6), (12,10), (16,14), (20,18)):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            fingers.append(1)      # 伸直记 1
        else:
            fingers.append(0)      # 弯曲记 0
    # 第二步：判断拇指（要结合左右手标签）
    if handedness_label == "Right":
        thumb_extended = landmarks[4].x > landmarks[3].x
    else:
        thumb_extended = landmarks[4].x < landmarks[3].x
    fingers.append(1 if thumb_extended else 0)
    # 第三步：把 5 个 0/1 加起来，就是张开的手指数
    return sum(fingers)
```

**原理一句话总结**：四指比"高低"（y 坐标），拇指比"左右"（x 坐标），
伸直记 1、弯曲记 0，最后求和。详细推导见第七章。

### 6.6 画骨架：`draw_hand_landmarks()`

```python
h, w = image.shape[:2]
# ① 把 21 个归一化坐标换算成像素坐标
points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

# ② 画 21 条骨架连线（绿色）
for start, end in HAND_CONNECTIONS:
    cv2.line(image, points[start], points[end], color=(0, 255, 0), thickness=2)

# ③ 画 21 个关键点
for idx, point in enumerate(points):
    if idx in FINGERTIP_IDS:
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)  # 指尖红
    else:
        cv2.circle(image, point, radius=4, color=(255, 0, 0), thickness=-1)  # 关节蓝
```

要点：

- 为什么先画线、后画点？这样圆点会盖住线条交叉处，画面更清晰；
- `thickness=-1` 表示**实心圆**（`-1` 是 OpenCV 的特殊约定，表示填充）；
- `color=(0, 255, 0)` 是 BGR 顺序，对应绿色。

### 6.7 写中文：`get_chinese_font()` + `draw_chinese_texts()`

`cv2.putText()` 不支持中文，所以流程是：**OpenCV 图像 → PIL 图像 → 写中文 → 转回 OpenCV**。

```python
# 1. 找一个系统中文字体（微软雅黑 → 黑体 → 宋体 依次尝试）
font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", font_size)

# 2. OpenCV(BGR) 转成 PIL(RGB)
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)

# 3. 写文字（fill 是 RGB 顺序，黄色 = (255,255,0)）
draw.text((x, y), text, font=font, fill=(255, 255, 0))

# 4. PIL 转回 OpenCV(BGR)
image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
```

字体用全局变量 `_chinese_font` 缓存，避免每帧重复加载字体文件（每帧加载会很卡）。

### 6.8 打开摄像头：`open_camera()`（容错）

```python
backends = [
    (camera_index, cv2.CAP_DSHOW),   # Windows 优先：DirectShow，启动快
    (camera_index, cv2.CAP_ANY),     # 兜底：让 OpenCV 自动选择后端
]
for idx, backend in backends:
    cap = cv2.VideoCapture(idx, backend)
    if cap.isOpened():
        ret, _ = cap.read()          # 不光要打开成功，还要试读一帧
        if ret:
            return cap
    cap.release()
return None
```

容错思路：

1. Windows 先用 DirectShow 后端尝试；
2. 失败再用默认后端；
3. 打开成功后还要**试读一帧**，确认摄像头真的能用；
4. 全部失败返回 `None`，主程序给出友好提示（被占用？驱动？编号？）。

### 6.9 主循环：`main()`

`main()` 把前面所有零件串起来。完整流程按顺序看：

```python
# ① 命令行参数
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--snapshot", type=str, default="")
args = parser.parse_args()

# ② 准备模型 → 创建检测器 → 打开摄像头
ensure_model()
landmarker = create_landmarker(num_hands=2)
cap = open_camera(args.camera)
```

主循环（这就是第一章流水线的代码实现）：

```python
while True:
    ret, frame = cap.read()                          # 1. 取一帧
    if not ret:                                      # 2. 取帧失败的容错
        fail_count += 1
        if fail_count >= 30:
            break
        continue
    frame = cv2.flip(frame, 1)                       # 3. 镜像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 4. BGR→RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, frame_index * 33)  # 5. 检测
    frame_index += 1

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        draw_hand_landmarks(frame, landmarks)        # 6. 画骨架
        label = handedness[0].category_name
        finger_count = count_fingers(landmarks, label)  # 7. 数手指
        text_items.append((f"...：{finger_count}", ...))  # 8. 准备文字

    frame = draw_chinese_texts(frame, text_items)    # 9. 写中文
    cv2.imshow(window_name, frame)                   # 10. 显示
    if cv2.waitKey(1) & 0xFF == ord('q'):            # 11. 按 q 退出
        break
```

关于 `cv2.waitKey(1) & 0xFF`：

- `waitKey(1)` 表示等待 1 毫秒按键，同时负责让窗口刷新；
- 它是**必须调用**的，否则 `imshow` 的窗口会卡死不刷新；
- `& 0xFF` 是为了兼容不同操作系统的按键编码，属于标准写法。

### 6.10 资源释放与异常处理

```python
except KeyboardInterrupt:
    print("收到中断信号，程序退出。")      # 终端 Ctrl+C 也能优雅退出
finally:
    cap.release()        # 释放摄像头
    landmarker.close()   # 关闭模型
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
```

`finally` 块无论程序正常结束、按 q 退出、还是发生异常，都会执行，
保证摄像头不会被"占用着不释放"。这是良好的编程习惯。

---

## 第七章 0~5 数字手势的判定原理

### 7.1 四根手指：比较"指尖和关节谁更靠上"

图像坐标系 y 向下。当手竖直、指尖朝上时：

- **手指伸直**：指尖在关节上方 → `指尖y < 关节y`
- **手指弯曲**：指尖蜷到关节下方 → `指尖y > 关节y`

```
       ● 指尖 TIP  y=0.30     伸直：0.30 < 0.55 ✓
       │
       ● PIP 关节 y=0.55

  弯曲时：
       ● PIP 关节 y=0.55
       │
       ● 指尖 TIP  y=0.70     弯曲：0.70 > 0.55 ✗
```

四根手指对应的"指尖、PIP 关节"编号：

| 手指 | 指尖 TIP | PIP 关节 | 伸直条件 |
| --- | --- | --- | --- |
| 食指 | 8 | 6 | `点8.y < 点6.y` |
| 中指 | 12 | 10 | `点12.y < 点10.y` |
| 无名指 | 16 | 14 | `点16.y < 点14.y` |
| 小指 | 20 | 18 | `点20.y < 点18.y` |

### 7.2 拇指：比较"指尖和关节谁更靠外"

拇指是横着长的，不能用 y 坐标判断，要用 **x 坐标**。

但左手和右手的拇指方向相反，怎么办？—— 借助 MediaPipe 给出的
**左右手标签（`Left` / `Right`）**。

因为画面做过镜像，在镜像画面里：

- 标签 `Right`：手在画面**右侧**，拇指朝**右**，伸直时指尖 x 更大
  → 条件：`点4.x > 点3.x`
- 标签 `Left`：手在画面**左侧**，拇指朝**左**，伸直时指尖 x 更小
  → 条件：`点4.x < 点3.x`

```
画面左侧（Left 标签）              画面右侧（Right 标签）
  4 3 2 ...（拇指朝左）              ... 2 3 4（拇指朝右）
伸直时 4 的 x < 3 的 x               伸直时 4 的 x > 3 的 x
```

### 7.3 五种数字手势的判定结果

| 手势 | 食指 | 中指 | 无名指 | 小指 | 拇指 | 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| 握拳 | 0 | 0 | 0 | 0 | 0 | **0** |
| 数字 1 | 1 | 0 | 0 | 0 | 0 | **1** |
| 数字 2 | 1 | 1 | 0 | 0 | 0 | **2** |
| 数字 3 | 1 | 1 | 1 | 0 | 0 | **3** |
| 数字 4 | 1 | 1 | 1 | 1 | 0 | **4** |
| 张开手掌 | 1 | 1 | 1 | 1 | 1 | **5** |

> 记忆口诀：**"四指看高低，拇指看左右；伸直记 1 分，弯曲记 0 分，五指求和就是数。"**

### 7.4 判定方法的局限（什么时候会不准）

了解边界，才能正确使用：

1. **手横过来时**：四指判断依赖 y 坐标（假设指尖朝上）。如果手指水平指向左右，y 比较会失效；
2. **手离镜头太近 / 关键点被遮挡**：模型定位不准，判断也会受影响；
3. **快速运动、光线很暗**：可能出现帧间抖动。

正确姿势：**手掌正对摄像头、指尖朝上、光线充足、手和背景区分明显**。

---

## 第八章 自测脚本怎么读（self_test.py）

### 8.1 为什么要写自测脚本

改完代码，怎么知道还能不能用？不能每次都靠"打开摄像头、伸手、肉眼看"。
`self_test.py` 可以**不需要摄像头画面窗口**就自动完成 5 类检查：

1. 依赖库能否导入；
2. 模型文件是否存在、能否加载；
3. 0~5 手指计数逻辑是否正确（用**假的关键点**做单元测试）；
4. 静态图片端到端推理（检测手、画骨架、数手指）；
5. 摄像头能否打开并读到一帧。

运行：

```bash
python self_test.py
```

### 8.2 怎么用假关键点做单元测试

这是初学者最值得学习的一个工程思想：**不依赖真实摄像头，也能测试核心逻辑**。

`count_fingers()` 只需要一个"带 x、y 属性的点列表"，并不关心这些点是不是模型算出来的。
于是可以用一个 `FakePoint` 类手工构造假数据：

```python
class FakePoint:
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y

# 构造一只"握拳"的虚拟手
# 食指、中指、无名指、小指的指尖 y=0.72（在 PIP 关节 y=0.60 下方 → 弯曲）
# 拇指指尖收在内侧 → 弯曲
# 期望 count_fingers() 返回 0
```

针对 0~5 每种手势都构造一组假数据，断言 `count_fingers()` 的返回值是否等于期望：
这就是"单元测试"的雏形。以后你修改了计数逻辑，跑一遍自测就知道有没有改坏。

### 8.3 静态图片测试

`assets/test_hands.jpg` 是 MediaPipe 官方示例图（含两只手）。
自测时用 `IMAGE` 模式对它推理，并把画了骨架的结果保存为
`assets/self_test_output.jpg`，方便肉眼检查。

---

## 第九章 动手练习（难度递增）

> 建议按顺序做，每题都给了提示，答案要靠自己改代码实现。

### 练习 1：改骨架颜色（★☆☆☆☆）

把骨架连线改成蓝色，指尖改成绿色。

- 提示：修改 `draw_hand_landmarks()` 里的 `color=(B, G, R)`，注意是 BGR 顺序。

### 练习 2：在画面上加一行自己的文字（★☆☆☆☆）

在画面顶部显示"我的第一个手势识别程序"。

- 提示：找到主循环里 `text_items.append(...)` 的地方，再加一行。

### 练习 3：识别"OK"手势（★★☆☆☆）

比"OK"时，拇指和食指捏成一个圈，其它三指伸直。

- 提示：判断拇指尖（4）和食指尖（8）之间的距离，距离很小且中指/无名指/小指伸直时识别为 OK。
- 两点距离公式：`math.sqrt((x1-x2)**2 + (y1-y2)**2)`。

### 练习 4：识别"比心 / 点赞"手势（★★☆☆☆）

- 提示：点赞时只有拇指竖直向上、其余四指弯曲；可以重点看点 4 的 y 坐标和点 8/12/16/20 的状态。

### 练习 5：用手势控制电脑（★★★☆☆）

识别到数字 5 时把系统音量调大，握拳（0）时静音。

- 提示：了解一下 `pyautogui` 库（可以模拟键盘按键）；在 `count_fingers` 结果变化时触发一次按键，注意"加锁"避免每秒触发几十次。

### 练习 6：识别一张静态图片（★★☆☆☆）

不打开摄像头，直接对一张照片做手势识别并保存结果。

- 提示：参考 `self_test.py` 里的静态图片测试部分，把检测器模式换成 `IMAGE`，用 `detect()` 方法。

### 练习 7：识别 3 只手（★★☆☆☆）

- 提示：把 `num_hands` 改成 3。

### 练习 8：提高/降低识别灵敏度（★☆☆☆☆）

- 提示：调整三个 confidence 阈值，观察画面里手被检测到的难易程度。

---

## 第十章 常见问题 FAQ

**Q1：提示"无法打开摄像头"？**

1. 摄像头是否被微信、腾讯会议、浏览器等占用，先关闭它们；
2. 检查系统"隐私设置"是否允许应用访问摄像头；
3. 外接摄像头试试 `python main.py --camera 1`。

**Q2：`pip install mediapipe` 安装失败？**

- 确认 Python 版本为 3.9~3.13；
- 换国内镜像源（见第三章）；
- 确认命令里包名是 `mediapipe`，不是 `mediapipe-hands`。

**Q3：运行报 `No module named 'cv2'` 或 `No module named 'mediapipe'`？**

说明库装到了别的 Python 环境。请确认你用的是项目虚拟环境：
`.venv\Scripts\python.exe main.py`，或先激活 `.venv` 再运行。

**Q4：模型报 "Unable to open file"？**

路径含中文时，部分版本的 MediaPipe 底层 C++ 无法读取文件。
本项目已用"字节流加载模型"规避。建议项目路径不要过深、不要含特殊符号。

**Q5：中文显示成问号/乱码？**

- OpenCV 的 `putText` 不支持中文，必须用 PIL 绘制（项目已实现）；
- 确认 Windows 字体目录存在 `msyh.ttc`（微软雅黑），Windows 系统一般自带。

**Q6：识别数字一直跳、不稳定？**

- 手掌正对镜头、指尖朝上；
- 光线充足，手离镜头 30~80 厘米；
- 动作放慢；可适当提高 `min_tracking_confidence`。

**Q7：程序运行很卡？**

- 降低分辨率：`cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`；
- 减少检测手数：`num_hands=1`；
- 字体只加载一次（项目已经做了缓存）。

**Q8：`detect_for_video` 报时间戳错误？**

视频模式要求时间戳严格递增，且不能重复。项目用"帧编号×33"保证递增。
不要传入可能重复的系统时间。

---

## 第十一章 进阶学习路线

当你把本项目和第九章练习都跑通后，可以按下面的方向继续：

1. **更多手势**：
   - 用关键点之间的**距离、角度**识别 OK、握拳、点赞、比心等；
   - 了解 MediaPipe 自带的 **GestureRecognizer**（内置识别 7 种手势）；
2. **自定义手势数据集**：
   - 采集每种手势的关键点坐标，训练 KNN / SVM 等简单分类器；
   - 可以参考 GitHub 上的 hand-gesture-recognition 项目；
3. **用手势控制应用**：
   - 结合 `pyautogui` 控制 PPT 翻页、音量、视频播放；
   - 结合树莓派控制硬件（灯、机械臂）；
4. **从 2D 走向 3D**：MediaPipe 还提供 33 个**人体姿态点**、468 个**面部点**，可以做健身计数、表情识别；
5. **工程化**：学习把模型打包成 exe（`pyinstaller`）、做成网页（Streamlit / Flask）。

---

## 附录 A 21 个关键点完整索引表

| 编号 | 常量名 | 部位 | 关节类型 |
| --- | --- | --- | --- |
| 0 | WRIST | 手腕 | — |
| 1 | THUMB_CMC | 拇指 | 腕掌关节 |
| 2 | THUMB_MCP | 拇指 | 掌指关节 |
| 3 | THUMB_IP | 拇指 | 指间关节 |
| 4 | THUMB_TIP | 拇指 | 指尖 |
| 5 | INDEX_MCP | 食指 | 掌指关节 |
| 6 | INDEX_PIP | 食指 | 近端关节 |
| 7 | INDEX_DIP | 食指 | 远端关节 |
| 8 | INDEX_TIP | 食指 | 指尖 |
| 9 | MIDDLE_MCP | 中指 | 掌指关节 |
| 10 | MIDDLE_PIP | 中指 | 近端关节 |
| 11 | MIDDLE_DIP | 中指 | 远端关节 |
| 12 | MIDDLE_TIP | 中指 | 指尖 |
| 13 | RING_MCP | 无名指 | 掌指关节 |
| 14 | RING_PIP | 无名指 | 近端关节 |
| 15 | RING_DIP | 无名指 | 远端关节 |
| 16 | RING_TIP | 无名指 | 指尖 |
| 17 | PINKY_MCP | 小指 | 掌指关节 |
| 18 | PINKY_PIP | 小指 | 近端关节 |
| 19 | PINKY_DIP | 小指 | 远端关节 |
| 20 | PINKY_TIP | 小指 | 指尖 |

## 附录 B 常用 OpenCV API 速查

| API | 作用 |
| --- | --- |
| `cv2.VideoCapture(index, backend)` | 打开摄像头/视频 |
| `cap.read()` | 读取一帧，返回 `(是否成功, 画面)` |
| `cap.release()` | 释放摄像头 |
| `cv2.cvtColor(img, code)` | 颜色空间转换（如 `COLOR_BGR2RGB`） |
| `cv2.flip(img, 1)` | 画面翻转（1 水平，0 垂直，-1 全部） |
| `cv2.line(img, p1, p2, color, thickness)` | 画直线 |
| `cv2.circle(img, center, radius, color, thickness)` | 画圆，`thickness=-1` 为实心 |
| `cv2.putText(img, text, org, font, scale, color)` | 写英文文字（不支持中文） |
| `cv2.imshow(name, img)` | 显示窗口 |
| `cv2.waitKey(ms)` | 等待按键（毫秒），返回按键码 |
| `cv2.destroyAllWindows()` | 关闭所有窗口 |
| `np.fromfile(path, dtype)` + `cv2.imdecode` | 读取含中文路径的图片 |
| `cv2.imencode` + `tofile` | 保存含中文路径的图片 |

命令行参数：

| 参数 | 作用 |
| --- | --- |
| `--camera 0` | 指定摄像头编号 |
| `--max-frames 100` | 处理 100 帧后自动退出 |
| `--snapshot assets/a.jpg` | 无界面保存一帧带标注画面后退出 |

## 附录 C 术语表

| 术语 | 含义 |
| --- | --- |
| 帧（Frame） | 视频中的一张画面，视频就是连续播放的帧 |
| 像素（Pixel） | 图像的最小单位，一个彩色像素含 B/G/R 三个值 |
| 分辨率 | 画面宽高的像素数，如 640×480 |
| 归一化坐标 | 把坐标缩放到 0~1 之间，与分辨率无关 |
| 帧率（FPS） | 每秒处理/显示的帧数，30 FPS 表示每秒 30 帧 |
| BGR / RGB | 颜色通道顺序，OpenCV 用 BGR，MediaPipe 用 RGB |
| 置信度（Confidence） | 模型对判断结果的把握程度，0~1 |
| 关键点（Landmark） | 模型输出的特征点坐标，本项目为 21 个手部关键点 |
| 虚拟环境（venv） | 项目专属的独立 Python 依赖环境 |
| 单元测试 | 用构造的数据自动验证某个函数是否正确 |
| Tasks API | MediaPipe 新一代推理接口（旧的 `solutions` 接口已废弃） |

---

> 📌 **学习建议收尾**：第一遍读不懂很正常，先跑通程序 → 再对照第六章看主循环 →
> 最后盯住第七章理解手指计数，你会发现整个项目其实就这一个核心算法。
> 然后完成第九章练习，你就能独立改造这个项目了。
