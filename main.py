# -*- coding: utf-8 -*-
"""
====================================================================
实时手势识别程序（OpenCV + MediaPipe HandLandmarker）
====================================================================
功能简介：
    1. 调用电脑摄像头实时读取视频流；
    2. 使用 MediaPipe 检测手部 21 个关键点，并绘制关键点与骨架连线；
    3. 根据“张开的手指数量”识别 0~5 的数字手势；
    4. 在视频窗口上实时显示识别结果；
    5. 按下键盘 q 键退出程序。

运行方式：
    python main.py              # 默认打开 0 号摄像头
    python main.py --camera 1   # 指定摄像头编号

作者备注：
    MediaPipe 1.x 版本已经移除了旧的 mp.solutions.hands 接口，
    本程序改用官方推荐的 Tasks API（HandLandmarker），
    需要搭配 models/hand_landmarker.task 模型文件使用（缺失会自动下载）。
====================================================================
"""

# ---------- 标准库 ----------
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ---------- 屏蔽 MediaPipe / TFLite 的冗余日志（必须在 import mediapipe 之前设置） ----------
# GLOG_minloglevel：0=INFO 1=WARNING 2=ERROR 3=FATAL，这里设为 2，只保留报错
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------- 第三方库 ----------
import cv2                      # 摄像头采集、图像处理、窗口显示
import numpy as np              # 图像数组处理
import mediapipe as mp          # MediaPipe 手部关键点检测
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw, ImageFont   # 用于在画面上绘制中文（OpenCV 原生不支持中文）


# ====================================================================
# 一、路径与模型配置
# ====================================================================

# 项目根目录（main.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent

# 手部关键点模型文件路径
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# 模型下载地址（官方地址；如果本地缺失会自动下载）
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task",
]


# ====================================================================
# 二、MediaPipe 21 个手部关键点索引（与官方定义一一对应）
# ====================================================================
#
#           0  手腕 WRIST
#          / \
#   拇指  1-2-3-4      （CMC-MCP-IP-TIP）
#   食指  5-6-7-8      （MCP-PIP-DIP-TIP）
#   中指  9-10-11-12
#   无名指 13-14-15-16
#   小指  17-18-19-20
#
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 五个指尖的索引集合（绘制时用醒目颜色）
FINGERTIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# 骨架连线关系表（21 个点之间的连接关系）
# 手掌：腕部与四根手指根部相连，各手指根部依次相连
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    (1, 2), (2, 3), (3, 4),          # 拇指
    (5, 6), (6, 7), (7, 8),          # 食指
    (9, 10), (10, 11), (11, 12),     # 中指
    (13, 14), (14, 15), (15, 16),    # 无名指
    (17, 18), (18, 19), (19, 20),    # 小指
]

# 数字手势 -> 中文说明
GESTURE_TEXT = {
    0: "数字 0（握拳）",
    1: "数字 1",
    2: "数字 2",
    3: "数字 3",
    4: "数字 4",
    5: "数字 5（张开手掌）",
}


# ====================================================================
# 三、中文路径兼容的图片读写工具函数
# ====================================================================
# OpenCV 原生 imread/imwrite 在 Windows 上不支持含中文的路径，
# 下面用 np.fromfile + imdecode / imencode + tofile 的方式替代。

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取图片（支持中文路径），失败返回 None。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, image):
    """保存图片（支持中文路径），成功返回 True。"""
    suffix = Path(path).suffix if isinstance(path, Path) else os.path.splitext(path)[1]
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


# ====================================================================
# 四、模型下载与检测器创建
# ====================================================================

def _download_progress(block_num, block_size, total_size):
    """urllib 下载进度回调：在控制台显示下载百分比"""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100)
    print(f"\r正在下载手部模型文件：{percent:5.1f}%", end="")
    if percent >= 100:
        print()  # 下载完成后换行


def ensure_model():
    """
    检查模型文件是否存在，不存在则自动下载。
    返回 True 表示模型就绪，False 表示获取失败。
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024:
        # 文件存在且大于 1MB，认为模型完整（模型实际约 7.8MB）
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("未找到手部模型文件，开始自动下载 ...")
    for url in MODEL_URLS:
        try:
            print(f"下载地址：{url}")
            urllib.request.urlretrieve(url, MODEL_PATH, reporthook=_download_progress)
            if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024:
                print("模型下载成功。")
                return True
        except Exception as exc:  # 网络异常、下载失败等
            print(f"\n该地址下载失败：{exc}")
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()  # 删除不完整的临时文件
    print("模型下载失败，请手动下载模型后放入 models 目录：")
    print(f"  目标路径：{MODEL_PATH}")
    print("  下载地址：" + MODEL_URLS[0])
    return False


def create_landmarker(num_hands=2):
    """
    创建并返回 HandLandmarker 手部关键点检测器。
    running_mode=VIDEO：视频流模式，需要为每帧提供单调递增的时间戳。

    注意：这里用 Python 读取模型字节流后通过 model_asset_buffer 传入，
    而不是直接传 model_asset_path。原因是模型底层的 C++ 文件接口
    在 Windows 上不支持中文路径（本项目路径包含“手势识别”），
    传字节流可以彻底规避该问题。
    """
    model_buffer = MODEL_PATH.read_bytes()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_buffer),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,                # 最多同时检测 2 只手
        min_hand_detection_confidence=0.5,  # 手部检测置信度阈值
        min_hand_presence_confidence=0.5,   # 手部存在置信度阈值
        min_tracking_confidence=0.5,        # 关键点跟踪置信度阈值
    )
    return vision.HandLandmarker.create_from_options(options)


# ====================================================================
# 五、手势识别逻辑
# ====================================================================

def count_fingers(landmarks, handedness_label):
    """
    根据手部关键点判断“张开的手指数量”，结果为 0~5。

    判断原理：
      - 食指/中指/无名指/小指：图像坐标系中 y 轴向下为正，
        当手指竖直伸直时，指尖（TIP）的 y 坐标会小于该手指 PIP 关节的 y 坐标；
        手指弯曲时，指尖会移动到 PIP 关节下方（y 更大）。
      - 拇指：比较拇指尖（4）与拇指 IP 关节（3）的 x 坐标。
        画面在进入检测器前已做水平镜像（自拍视角），
        标签为 "Right" 的手位于画面右侧、拇指朝右，伸直时指尖 x 更大；
        标签为 "Left"  的手位于画面左侧、拇指朝左，伸直时指尖 x 更小。

    参数：
        landmarks        : MediaPipe 返回的 21 个关键点列表，每个点含 x、y（归一化坐标）
        handedness_label : "Left" 或 "Right"
    返回：
        张开的手指数量，整数 0~5
    """
    fingers = []

    # 1) 四根手指：食指、中指、无名指、小指，每组 = (指尖索引, PIP 关节索引)
    for tip_id, pip_id in (
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            fingers.append(1)  # 伸直
        else:
            fingers.append(0)  # 弯曲

    # 2) 拇指：根据左右手标签判断 x 坐标的比较方向
    if handedness_label == "Right":
        # 拇指在画面右侧，伸直时指尖 x 大于 IP 关节 x
        thumb_extended = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
    else:
        # 拇指在画面左侧，伸直时指尖 x 小于 IP 关节 x
        thumb_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    fingers.append(1 if thumb_extended else 0)

    return sum(fingers)


# ====================================================================
# 六、画面绘制（骨架、关键点、中文文字）
# ====================================================================

# 中文字体缓存（避免每帧重复加载字体）
_chinese_font = None


def get_chinese_font(font_size=30):
    """加载 Windows 系统中文字体（微软雅黑 / 黑体 / 宋体依次尝试）。"""
    global _chinese_font
    if _chinese_font is not None:
        return _chinese_font
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",  # 黑体
        r"C:\Windows\Fonts\simsun.ttc",  # 宋体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux 兜底
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                _chinese_font = ImageFont.truetype(font_path, font_size)
                return _chinese_font
            except Exception:
                continue
    return None


def draw_hand_landmarks(image, landmarks):
    """
    在图像上绘制 21 个手部关键点与骨架连线。
    :param image:     OpenCV 图像（BGR）
    :param landmarks: 21 个关键点（归一化坐标）
    """
    h, w = image.shape[:2]
    # 将归一化坐标（0~1）换算成像素坐标
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # 先画骨架连线（绿色）
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], color=(0, 255, 0), thickness=2)

    # 再画关键点（普通关节为蓝色实心圆，指尖为红色实心圆，方便观察）
    for idx, point in enumerate(points):
        if idx in FINGERTIP_IDS:
            cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)
        else:
            cv2.circle(image, point, radius=4, color=(255, 0, 0), thickness=-1)


def draw_chinese_texts(image, text_items):
    """
    在画面上一次性绘制多段中文文字（OpenCV 自带 putText 不支持中文，
    因此借助 PIL 绘制后再转回 OpenCV 图像）。
    :param image:     OpenCV 图像（BGR）
    :param text_items: 列表，每个元素为 (文字内容, (x, y), 字号)
    :return: 绘制后的 OpenCV 图像
    """
    font = get_chinese_font()
    if font is None:
        # 没有可用中文字体时，用英文兜底（避免程序崩溃）
        for text, (x, y), _ in text_items:
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
        return image

    # BGR -> RGB -> PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for text, (x, y), font_size in text_items:
        try:
            item_font = font if font_size == 30 else ImageFont.truetype(
                r"C:\Windows\Fonts\msyh.ttc", font_size)
        except Exception:
            item_font = font
        # PIL 使用 RGB 颜色，OpenCV 是 BGR，这里统一使用黄色 (255,255,0)
        draw.text((x, y), text, font=item_font, fill=(255, 255, 0))
    # PIL 图像 -> RGB -> BGR
    image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return image


# ====================================================================
# 七、摄像头打开（带容错）
# ====================================================================

def open_camera(camera_index=0):
    """
    尝试打开摄像头。
    Windows 下优先使用 DirectShow 后端（启动快、兼容性好），
    失败后再尝试默认后端，最大限度提高容错。
    :return: 成功返回 cv2.VideoCapture，失败返回 None
    """
    # Windows 有 CAP_DSHOW；其它平台该常量虽存在但不生效，统一放在尝试列表中
    backends = [
        (camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else None,
        (camera_index, cv2.CAP_ANY),
    ]
    backends = [item for item in backends if item is not None]

    for idx, backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # 尝试读取一帧，确认摄像头真的可用
                ret, _ = cap.read()
                if ret:
                    print(f"摄像头 {idx} 打开成功（后端：{backend}）")
                    return cap
            cap.release()
        except Exception as exc:
            print(f"尝试打开摄像头 {idx}（后端 {backend}）时出现异常：{exc}")
    return None


# ====================================================================
# 八、主程序
# ====================================================================

def main():
    # 1) 解析命令行参数
    parser = argparse.ArgumentParser(description="实时手势识别（0-5 数字手势）")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头编号，默认 0")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧后自动退出，0 表示不限制（默认 0）")
    parser.add_argument("--snapshot", type=str, default="",
                        help="无界面自测用：保存一帧带标注的画面到指定路径后退出，"
                             "例如 --snapshot assets/snapshot.jpg")
    args = parser.parse_args()
    headless = bool(args.snapshot)  # 保存截图模式下不弹出窗口，方便无人值守自测

    # 2) 检查并准备模型
    if not ensure_model():
        sys.exit(1)

    # 3) 创建检测器（创建失败时给出明确提示）
    try:
        landmarker = create_landmarker(num_hands=2)
    except Exception as exc:
        print(f"手部检测器创建失败：{exc}")
        sys.exit(1)

    # 4) 打开摄像头
    print("正在打开摄像头 ...")
    cap = open_camera(args.camera)
    if cap is None:
        print("无法打开摄像头，请检查：")
        print("  1) 摄像头是否被其它程序（如会议软件）占用；")
        print("  2) 摄像头驱动是否正常、系统隐私设置是否允许应用访问摄像头；")
        print("  3) 若使用外接摄像头，可尝试指定编号：python main.py --camera 1")
        landmarker.close()
        sys.exit(1)

    window_name = "Hand Gesture Recognition (press q to quit)"
    if not headless:
        # 正常模式才创建显示窗口；截图自测模式无窗口运行
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_index = 0       # 帧序号，同时用于生成单调递增的时间戳
    fail_count = 0        # 连续读帧失败计数
    MAX_FAIL_FRAMES = 30  # 允许的最大连续失败帧数
    SNAPSHOT_WARMUP = 10  # 截图模式下先跑若干帧，等待检测稳定

    print("开始识别：把手举到摄像头前，握拳=0，伸手指=1~5；按 q 键退出。")

    try:
        while True:
            # 5) 读取一帧视频
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                print(f"读取摄像头画面失败（第 {fail_count} 次）")
                if fail_count >= MAX_FAIL_FRAMES:
                    print("连续多次读取失败，摄像头可能已断开，程序退出。")
                    break
                cv2.waitKey(30)
                continue
            fail_count = 0

            # 6) 水平镜像，变成“自拍视角”，举手方向与画面一致，体验更自然
            frame = cv2.flip(frame, 1)

            # 7) BGR -> RGB，包装成 MediaPipe 图像格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 8) 关键点检测（VIDEO 模式要求时间戳严格递增，单位毫秒）
            timestamp_ms = frame_index * 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            # 9) 遍历检测到的手，画骨架并统计手指数量
            text_items = []
            for hand_idx, (landmarks, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):
                draw_hand_landmarks(frame, landmarks)

                # handedness 是一个分类列表，取置信度最高的那一项
                label = handedness[0].category_name
                finger_count = count_fingers(landmarks, label)
                gesture_text = GESTURE_TEXT.get(finger_count, str(finger_count))

                # 在手腕附近显示这只手的识别结果
                h, w = frame.shape[:2]
                wrist = landmarks[WRIST]
                text_x = max(10, int(wrist.x * w) - 60)
                text_y = max(35, int(wrist.y * h) - 20)
                text_items.append((f"{hand_idx + 1}号手：{gesture_text}",
                                   (text_x, text_y), 28))

            # 10) 顶部固定提示信息
            if result.hand_landmarks:
                text_items.append(("实时手势识别 - 按 q 键退出", (10, 10), 24))
            else:
                text_items.append(("未检测到手部，请把手放进画面", (10, 10), 24))

            # 11) 统一绘制中文文字
            frame = draw_chinese_texts(frame, text_items)

            # 12) 无界面自测模式：预热若干帧后保存一帧带标注画面并退出
            if headless:
                if frame_index >= SNAPSHOT_WARMUP or result.hand_landmarks:
                    saved = imwrite_unicode(args.snapshot, frame)
                    if saved:
                        print(f"已保存自测截图：{args.snapshot}")
                    else:
                        print(f"截图保存失败：{args.snapshot}")
                    break
            else:
                # 13) 正常模式：显示画面
                cv2.imshow(window_name, frame)

                # 按键检测：按 q 退出（waitKey 必须调用，否则窗口不刷新）
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("检测到 q 键，程序退出。")
                    break

            # 14) 限制最大帧数（便于自动化测试）
            if args.max_frames > 0 and frame_index >= args.max_frames:
                print(f"已达到最大帧数 {args.max_frames}，自动退出。")
                break

    except KeyboardInterrupt:
        # 终端里按 Ctrl+C 也能优雅退出
        print("收到中断信号，程序退出。")
    finally:
        # 15) 释放摄像头、关闭检测器和所有窗口
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print("资源已释放，程序结束。")


if __name__ == "__main__":
    main()
