# -*- coding: utf-8 -*-
"""
====================================================================
项目自测脚本（无需摄像头画面窗口，适合首次运行前检查环境）
====================================================================
检查项目：
    1. 第三方依赖库是否正确安装；
    2. 手部模型文件是否存在、能否正常加载；
    3. 0~5 数字手势识别逻辑是否正确（使用构造的关键点做单元测试）；
    4. 静态图片端到端推理（检测手部 -> 画骨架 -> 统计手指数量）；
    5. 摄像头能否正常打开并读取一帧（没有摄像头会标记为 SKIP，不影响其它结果）。

运行方式：
    python self_test.py
====================================================================
"""

import os
import sys
import traceback
from pathlib import Path

# 屏蔽 MediaPipe 冗余日志
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

# 导入主程序中的函数与常量（main.py 与本文件在同一目录）
import main
from main import (
    MODEL_PATH,
    WRIST,
    THUMB_IP, THUMB_TIP,
    INDEX_TIP, INDEX_PIP,
    MIDDLE_TIP, MIDDLE_PIP,
    RING_TIP, RING_PIP,
    PINKY_TIP, PINKY_PIP,
    count_fingers,
    draw_hand_landmarks,
    draw_chinese_texts,
    ensure_model,
    open_camera,
)

# 统计测试结果
PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def check(name, ok, detail=""):
    """打印单项测试结果，并累计计数。"""
    global PASS_COUNT, FAIL_COUNT
    if ok:
        PASS_COUNT += 1
        print(f"[PASS] {name} {detail}")
    else:
        FAIL_COUNT += 1
        print(f"[FAIL] {name} {detail}")


def skip(name, detail=""):
    """标记跳过项（例如没有摄像头）。"""
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"[SKIP] {name} {detail}")


# --------------------------------------------------------------------
# 1. 依赖库导入检查
# --------------------------------------------------------------------
def test_imports():
    print("\n========== 1. 依赖库检查 ==========")
    try:
        import mediapipe as mp
        check("导入 mediapipe", True, f"（版本 {mp.__version__}）")
    except Exception as exc:
        check("导入 mediapipe", False, f"（{exc}）")
        return
    check("导入 OpenCV (cv2)", True, f"（版本 {cv2.__version__}）")
    check("导入 numpy", True, f"（版本 {np.__version__}）")
    try:
        from PIL import Image  # noqa: F401
        check("导入 Pillow (PIL)", True, "（用于中文文字绘制）")
    except Exception as exc:
        check("导入 Pillow (PIL)", False, f"（{exc}）")
    try:
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: F401
        check("导入 MediaPipe Tasks API", True)
    except Exception as exc:
        check("导入 MediaPipe Tasks API", False, f"（{exc}）")


# --------------------------------------------------------------------
# 2. 模型文件检查
# --------------------------------------------------------------------
def test_model():
    print("\n========== 2. 模型文件检查 ==========")
    exists = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024 * 1024
    check("模型文件存在且大小正常", exists,
          f"（路径：{MODEL_PATH}，大小："
          f"{MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB）" if exists else "")

    landmarker = None
    try:
        if not exists:
            # 尝试自动下载
            check("自动下载模型", ensure_model())
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        # 用字节流方式加载，规避 Windows 中文路径打不开模型文件的问题
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        check("模型加载并创建检测器", True)
    except Exception as exc:
        check("模型加载并创建检测器", False, f"（{exc}）")
    return landmarker


# --------------------------------------------------------------------
# 3. 手指计数逻辑单元测试（构造虚拟关键点，不依赖真实手部）
# --------------------------------------------------------------------
class FakePoint:
    """模拟 MediaPipe 关键点：只包含归一化 x、y 坐标。"""
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y


def make_hand(index_open, middle_open, ring_open, pinky_open,
              thumb_open, label="Right"):
    """
    构造一只虚拟手的 21 个关键点。
    四个布尔参数控制四根手指伸直/弯曲；
    label='Right' 时拇指位于画面右侧（伸直时 x 更大），
    label='Left'  时拇指位于画面左侧（伸直时 x 更小）。
    """
    pts = [FakePoint(0.5, 0.90) for _ in range(21)]  # 先给所有点一个默认坐标

    # 四根手指：(指尖, PIP关节, 该手指的水平位置)
    # 伸直时指尖 y=0.35（在 PIP 上方），弯曲时 y=0.72（在 PIP 下方）
    finger_defs = [
        (INDEX_TIP,  INDEX_PIP,  0.35, index_open),
        (MIDDLE_TIP, MIDDLE_PIP, 0.45, middle_open),
        (RING_TIP,   RING_PIP,   0.55, ring_open),
        (PINKY_TIP,  PINKY_PIP,  0.65, pinky_open),
    ]
    for tip, pip, x, is_open in finger_defs:
        pts[pip] = FakePoint(x, 0.60)
        pts[tip] = FakePoint(x, 0.35 if is_open else 0.72)

    # 拇指：IP 关节固定，伸直时指尖向手外侧移动，弯曲时收回掌心一侧
    pts[THUMB_IP] = FakePoint(0.50, 0.55)
    if label == "Right":
        # 画面右侧的拇指：伸直时 x 更大（0.72），弯曲时收回至 x 小于 IP
        pts[THUMB_TIP] = FakePoint(0.72, 0.55) if thumb_open else FakePoint(0.48, 0.55)
    else:
        # 画面左侧的拇指：伸直时 x 更小（0.28），弯曲时收回至 x 大于 IP
        pts[THUMB_TIP] = FakePoint(0.28, 0.55) if thumb_open else FakePoint(0.52, 0.55)
    return pts


def test_finger_count_logic():
    print("\n========== 3. 手指计数逻辑单元测试 ==========")
    cases = [
        # 名称, 食/中/无名/小/拇指, 标签, 期望数量
        ("握拳（0）",   (False, False, False, False, False), "Right", 0),
        ("食指（1）",   (True,  False, False, False, False), "Right", 1),
        ("两指（2）",   (True,  True,  False, False, False), "Right", 2),
        ("三指（3）",   (True,  True,  True,  False, False), "Right", 3),
        ("四指（4）",   (True,  True,  True,  True,  False), "Right", 4),
        ("五指（5）",   (True,  True,  True,  True,  True),  "Right", 5),
        ("仅拇指（1）", (False, False, False, False, True),  "Right", 1),
        ("五指-左手镜像", (True, True, True, True, True),    "Left", 5),
    ]
    for name, flags, label, expected in cases:
        pts = make_hand(*flags, label=label)
        got = count_fingers(pts, label)
        check(f"手势 {name}", got == expected, f"（期望 {expected}，实际 {got}）")


# --------------------------------------------------------------------
# 4. 静态图片端到端推理测试
# --------------------------------------------------------------------
def test_image_inference(landmarker):
    print("\n========== 4. 静态图片端到端推理测试 ==========")
    if landmarker is None:
        check("静态图片推理", False, "（检测器未创建，跳过）")
        return

    import mediapipe as mp

    image_path = main.BASE_DIR / "assets" / "test_hands.jpg"
    if not image_path.exists():
        check("读取测试图片", False, f"（未找到 {image_path}）")
        return

    # 用支持中文路径的方式读取图片
    image = main.imread_unicode(image_path)
    check("读取测试图片", image is not None, f"（{image_path.name}）")

    # 为了和主程序保持一致，使用镜像（自拍视角）后的画面
    mirrored = cv2.flip(image, 1)
    rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    hand_num = len(result.hand_landmarks)
    check("检测到 2 只手", hand_num == 2, f"（实际检测到 {hand_num} 只）")

    text_items = []
    for idx, (landmarks, handedness) in enumerate(
            zip(result.hand_landmarks, result.handedness)):
        draw_hand_landmarks(mirrored, landmarks)
        label = handedness[0].category_name
        finger_count = count_fingers(landmarks, label)
        h, w = mirrored.shape[:2]
        x = int(landmarks[WRIST].x * w)
        y = int(landmarks[WRIST].y * h)
        text_items.append((f"{idx + 1}号手（{label}）：{finger_count}",
                           (max(10, x - 80), max(30, y - 20)), 28))
        print(f"  - {idx + 1} 号手：左右手标签={label}，张开手指数={finger_count}")

    text_items.append(("静态图片自测结果", (10, 10), 26))
    mirrored = draw_chinese_texts(mirrored, text_items)

    output_path = main.BASE_DIR / "assets" / "self_test_output.jpg"
    saved = main.imwrite_unicode(output_path, mirrored)
    check("保存渲染结果图", saved and output_path.exists(), f"（{output_path}）")

    landmarker.close()


# --------------------------------------------------------------------
# 5. 摄像头快速测试（只打开并读取一帧，不弹窗）
# --------------------------------------------------------------------
def test_camera():
    print("\n========== 5. 摄像头检查 ==========")
    cap = None
    try:
        cap = open_camera(0)
        if cap is None:
            skip("摄像头", "（未检测到可用摄像头；若使用外接摄像头，"
                          "可尝试 python main.py --camera 1）")
            return
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            check("摄像头读取一帧", True,
                  f"（画面尺寸：{frame.shape[1]} x {frame.shape[0]}）")
        else:
            check("摄像头读取一帧", False)
    except Exception as exc:
        check("摄像头", False, f"（异常：{exc}）")
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()


# --------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------
def main_test():
    print("=" * 60)
    print("Python 手势识别项目 - 自测开始")
    print("=" * 60)
    print("Python 版本：", sys.version.split()[0])
    print("项目目录：", main.BASE_DIR)

    test_imports()
    landmarker = test_model()
    test_finger_count_logic()
    test_image_inference(landmarker)
    test_camera()

    print("\n" + "=" * 60)
    print(f"自测结束：通过 {PASS_COUNT} 项，失败 {FAIL_COUNT} 项，跳过 {SKIP_COUNT} 项")
    if FAIL_COUNT == 0:
        print("全部检查通过，可以运行：python main.py")
    else:
        print("存在失败项，请根据上方提示排查后再运行主程序。")
    print("=" * 60)
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main_test()
