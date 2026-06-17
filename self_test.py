"""不依赖 ComfyUI 的最小自测脚本。

读取 `测试样本/` 里的图片和 mask，构造 3 组动画 JSON，
调用 `DecorAnimationPlayer` 的核心动画与编码流程，并把结果写到
`测试样本/selftest_output.mp4`。
"""

import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

# 让脚本能找到当前目录下的 nodes.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import nodes  # noqa: E402


SAMPLE_DIR = os.path.join(CURRENT_DIR, "测试样本")
IMAGE_PATH = os.path.join(SAMPLE_DIR, "iamge.png")
MASK_PATH = os.path.join(SAMPLE_DIR, "mask.png")
BASE_PATH = os.path.join(SAMPLE_DIR, "base_image.png")
OUTPUT_PATH = os.path.join(SAMPLE_DIR, "selftest_output.mp4")


def load_image_as_tensor(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)  # [1, H, W, 3]
    return tensor


def load_mask_as_tensor(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("L")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)  # [1, H, W]
    return tensor


def main() -> int:
    print("[selftest] start")
    print(f"[selftest] sample dir: {SAMPLE_DIR}")

    if not os.path.isdir(SAMPLE_DIR):
        print("[selftest] FAIL: 测试样本 目录不存在")
        return 1

    for label, path in [
        ("image", IMAGE_PATH),
        ("mask", MASK_PATH),
        ("base_image", BASE_PATH),
    ]:
        if not os.path.isfile(path):
            print(f"[selftest] FAIL: 缺少 {label} 文件: {path}")
            return 1
        with Image.open(path) as img:
            print(f"[selftest] {label}: {path} -> size={img.size} mode={img.mode}")

    image_tensor = load_image_as_tensor(IMAGE_PATH)
    mask_tensor = load_mask_as_tensor(MASK_PATH)
    base_tensor = load_image_as_tensor(BASE_PATH)
    animation_json_1 = """{
  "fps": 12,
  "frame_count": 24,
  "translate_y": [
    { "frame": 0, "value": 0, "easing": "ease_in_out" },
    { "frame": 12, "value": -36, "easing": "ease_out" },
    { "frame": 23, "value": 0, "easing": "ease_in" }
  ],
  "scale": [
    { "frame": 0, "value": 1.0, "easing": "ease_in_out" },
    { "frame": 12, "value": 1.12, "easing": "ease_in_out" },
    { "frame": 23, "value": 1.0, "easing": "ease_in_out" }
  ],
  "rotation": [
    { "frame": 0, "value": -8, "easing": "ease_in_out" },
    { "frame": 12, "value": 8, "easing": "ease_in_out" },
    { "frame": 23, "value": -8, "easing": "ease_in_out" }
  ],
  "opacity": 0.9
}"""
    animation_json_2 = """{
  "fps": 12,
  "frame_count": 24,
  "translate_x": [
    { "frame": 0, "value": -48, "easing": "ease_in_out" },
    { "frame": 23, "value": 48, "easing": "ease_in_out" }
  ],
  "scale": {
    "from": 0.88,
    "to": 1.0,
    "easing": "ease_in_out"
  },
  "rotation": 0,
  "opacity": 0.55
}"""
    animation_json_3 = """{
  "fps": 12,
  "frame_count": 24,
  "translate_y": [
    { "frame": 0, "value": 28, "easing": "ease_in_out" },
    { "frame": 12, "value": 10, "easing": "ease_out" },
    { "frame": 23, "value": 28, "easing": "ease_in" }
  ],
  "scale": [
    { "frame": 0, "value": 0.78, "easing": "ease_in_out" },
    { "frame": 12, "value": 0.86, "easing": "ease_in_out" },
    { "frame": 23, "value": 0.78, "easing": "ease_in_out" }
  ],
  "rotation": [
    { "frame": 0, "value": 10, "easing": "ease_in_out" },
    { "frame": 23, "value": -10, "easing": "ease_in_out" }
  ],
  "opacity": 0.45
}"""

    try:
        result = nodes.DecorAnimationPlayer().animate(
            image_1=image_tensor,
            mask_1=mask_tensor,
            animation_json_1=animation_json_1,
            image_2=image_tensor,
            mask_2=mask_tensor,
            animation_json_2=animation_json_2,
            image_3=image_tensor,
            mask_3=mask_tensor,
            animation_json_3=animation_json_3,
            base_image=base_tensor,
            fps_override=0,
            filename_prefix="selftest/decor_animation",
            mp4_crf=20,
        )
    except Exception:  # pragma: no cover
        print("[selftest] FAIL: animate() 抛出异常")
        traceback.print_exc()
        return 2

    ui_payload = result.get("ui", {})
    videos = ui_payload.get("videos", [])
    text_lines = ui_payload.get("text", [])

    if not videos:
        print("[selftest] FAIL: 返回结构中没有 videos")
        return 3

    video_meta = videos[0]
    video_path = video_meta.get("absolute_path") or video_meta.get("file_url") or ""
    print("[selftest] ui.videos[0]:", json.dumps(video_meta, ensure_ascii=False))
    for line in text_lines:
        print("[selftest] ui.text:", line)

    if video_path.startswith("file:///"):
        file_check = video_path[len("file:///") :]
    else:
        file_check = video_path
    file_check = file_check.replace("/", os.sep)

    if not os.path.isfile(file_check):
        print(f"[selftest] FAIL: 输出文件不存在: {file_check}")
        return 4

    size = os.path.getsize(file_check)
    print(f"[selftest] output file: {file_check} ({size} bytes)")

    if size <= 0:
        print("[selftest] FAIL: 输出文件大小为 0")
        return 5

    with open(file_check, "rb") as handle:
        head = handle.read(12)
    if not (head.startswith(b"\x00\x00\x00") and b"ftyp" in head[:12]):
        # mp4 的 ftyp 标志通常在偏移 4，不再严格要求精确位置
        if b"ftyp" not in head:
            print(f"[selftest] FAIL: 输出文件不是有效的 mp4, head={head!r}")
            return 6

    # 拷贝一份到 测试样本/selftest_output.mp4，方便用户直接打开
    try:
        with open(file_check, "rb") as src, open(OUTPUT_PATH, "wb") as dst:
            dst.write(src.read())
        print(f"[selftest] copied preview to: {OUTPUT_PATH}")
    except OSError as exc:
        print(f"[selftest] WARN: 复制预览文件失败: {exc}")

    print("[selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
