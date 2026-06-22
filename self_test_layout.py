"""不依赖 ComfyUI 的静态贴纸排版自测脚本。

读取 `测试样本/` 下的底图、3 张贴纸和对应 mask，再读取 3 份位置 JSON，
调用 `DecorStickerLayoutComposer` 的核心逻辑，把结果保存到 `测试样本/`。
"""

import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import nodes  # noqa: E402


SAMPLE_DIR = os.path.join(CURRENT_DIR, "测试样本")
BASE_PATH = os.path.join(SAMPLE_DIR, "base_image.png")
STICKER_PATHS = [
    os.path.join(SAMPLE_DIR, "装饰元素1.png"),
    os.path.join(SAMPLE_DIR, "装饰元素2.png"),
    os.path.join(SAMPLE_DIR, "装饰元素3.png"),
]
MASK_PATHS = [
    os.path.join(SAMPLE_DIR, "装饰元素1对应的mask.png"),
    os.path.join(SAMPLE_DIR, "装饰元素2对应的mask.png"),
    os.path.join(SAMPLE_DIR, "装饰元素3对应的mask.png"),
]
POSITION_PATHS = [
    os.path.join(SAMPLE_DIR, "position_1.json"),
    os.path.join(SAMPLE_DIR, "position_2.json"),
    os.path.join(SAMPLE_DIR, "position_3.json"),
]
OUTPUT_IMAGE_PATH = os.path.join(SAMPLE_DIR, "layout_test_output.png")
OUTPUT_MASK_PATHS = [
    os.path.join(SAMPLE_DIR, "layout_test_mask_1.png"),
    os.path.join(SAMPLE_DIR, "layout_test_mask_2.png"),
    os.path.join(SAMPLE_DIR, "layout_test_mask_3.png"),
]


def load_image_as_tensor(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def load_mask_as_tensor(path: str) -> torch.Tensor:
    pil = Image.open(path).convert("L")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def save_image_tensor(tensor: torch.Tensor, path: str) -> None:
    pil = nodes._tensor_image_to_pil(tensor)
    pil.save(path)


def save_mask_tensor(tensor: torch.Tensor, path: str) -> None:
    mask = nodes._tensor_mask_to_pil(tensor, (tensor.shape[-1], tensor.shape[-2]))
    mask.save(path)


def main() -> int:
    print("[layout-selftest] start")
    print(f"[layout-selftest] sample dir: {SAMPLE_DIR}")

    required_paths = [BASE_PATH, *STICKER_PATHS, *MASK_PATHS, *POSITION_PATHS]
    for path in required_paths:
        if not os.path.isfile(path):
            print(f"[layout-selftest] FAIL: 缺少文件: {path}")
            return 1

    for path in [BASE_PATH, *STICKER_PATHS, *MASK_PATHS]:
        with Image.open(path) as img:
            print(f"[layout-selftest] image: {os.path.basename(path)} size={img.size} mode={img.mode}")

    position_payloads = []
    for path in POSITION_PATHS:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        position_payloads.append(json.dumps(payload, ensure_ascii=False))
        print(f"[layout-selftest] position: {os.path.basename(path)} -> {payload}")

    try:
        image, mask_1, mask_2, mask_3 = nodes.DecorStickerLayoutComposer().compose(
            sticker_1=load_image_as_tensor(STICKER_PATHS[0]),
            mask_1=load_mask_as_tensor(MASK_PATHS[0]),
            position_json_1=position_payloads[0],
            sticker_2=load_image_as_tensor(STICKER_PATHS[1]),
            mask_2=load_mask_as_tensor(MASK_PATHS[1]),
            position_json_2=position_payloads[1],
            sticker_3=load_image_as_tensor(STICKER_PATHS[2]),
            mask_3=load_mask_as_tensor(MASK_PATHS[2]),
            position_json_3=position_payloads[2],
            base_image=load_image_as_tensor(BASE_PATH),
        )
    except Exception:
        print("[layout-selftest] FAIL: compose() 抛出异常")
        traceback.print_exc()
        return 2

    save_image_tensor(image, OUTPUT_IMAGE_PATH)
    for tensor, path in zip((mask_1, mask_2, mask_3), OUTPUT_MASK_PATHS):
        save_mask_tensor(tensor, path)

    for path in [OUTPUT_IMAGE_PATH, *OUTPUT_MASK_PATHS]:
        if not os.path.isfile(path):
            print(f"[layout-selftest] FAIL: 输出文件不存在: {path}")
            return 3
        print(f"[layout-selftest] output: {path} ({os.path.getsize(path)} bytes)")

    print("[layout-selftest] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
