import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Sequence, Tuple

import av
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:  # pragma: no cover
    folder_paths = None

try:
    import imageio_ffmpeg as _imageio_ffmpeg
except ImportError:  # pragma: no cover
    _imageio_ffmpeg = None


def _first_tensor_item(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 4:
        return value[0]
    return value


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    image = _first_tensor_item(image).detach().cpu().clamp(0.0, 1.0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _tensor_mask_to_pil(mask: torch.Tensor, size: Sequence[int]) -> Image.Image:
    mask = _first_tensor_item(mask).detach().cpu().clamp(0.0, 1.0).numpy()
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    mask = (mask * 255.0).round().astype(np.uint8)
    pil_mask = Image.fromarray(mask, mode="L")
    if pil_mask.size != tuple(size):
        pil_mask = pil_mask.resize(tuple(size), Image.Resampling.BILINEAR)
    return pil_mask


@dataclass
class Keyframe:
    frame: int
    value: float
    easing: str = "linear"


def _coerce_keyframes(raw_value: Any, last_frame: int, default: float) -> List[Keyframe]:
    if raw_value is None:
        return [Keyframe(0, float(default)), Keyframe(last_frame, float(default))]

    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        return [Keyframe(0, value), Keyframe(last_frame, value)]

    if isinstance(raw_value, dict):
        if "value" in raw_value and not any(k in raw_value for k in ("keyframes", "frames", "keys")):
            value = float(raw_value["value"])
            return [Keyframe(0, value), Keyframe(last_frame, value)]
        if "from" in raw_value or "to" in raw_value:
            start_value = float(raw_value.get("from", default))
            end_value = float(raw_value.get("to", start_value))
            easing = str(raw_value.get("easing", "linear"))
            return [Keyframe(0, start_value, easing), Keyframe(last_frame, end_value, easing)]
        raw_value = raw_value.get("keyframes") or raw_value.get("frames") or raw_value.get("keys")

    if not isinstance(raw_value, list) or not raw_value:
        return [Keyframe(0, float(default)), Keyframe(last_frame, float(default))]

    keyframes: List[Keyframe] = []
    for index, item in enumerate(raw_value):
        if isinstance(item, (int, float)):
            frame = int(round(index * last_frame / max(len(raw_value) - 1, 1)))
            keyframes.append(Keyframe(frame, float(item)))
            continue

        if not isinstance(item, dict):
            continue

        raw_frame = item.get("frame")
        if raw_frame is None:
            raw_time = item.get("time")
            raw_frame = index if raw_time is None else raw_time

        value = item.get("value", item.get("v", default))
        easing = str(item.get("easing", item.get("ease", "linear")))
        keyframes.append(Keyframe(int(round(float(raw_frame))), float(value), easing))

    if not keyframes:
        return [Keyframe(0, float(default)), Keyframe(last_frame, float(default))]

    keyframes.sort(key=lambda item: item.frame)
    if keyframes[0].frame > 0:
        keyframes.insert(0, Keyframe(0, keyframes[0].value, keyframes[0].easing))
    if keyframes[-1].frame < last_frame:
        keyframes.append(Keyframe(last_frame, keyframes[-1].value, keyframes[-1].easing))
    return keyframes


def _apply_easing(t: float, easing: str) -> float:
    easing = easing.lower()
    if easing in {"linear", "none"}:
        return t
    if easing in {"ease_in", "ease-in", "in"}:
        return t * t
    if easing in {"ease_out", "ease-out", "out"}:
        return 1.0 - (1.0 - t) * (1.0 - t)
    if easing in {"ease_in_out", "ease-in-out", "in_out"}:
        return 3.0 * t * t - 2.0 * t * t * t
    if easing == "sin":
        return math.sin((t * math.pi) / 2.0)
    return t


def _sample_keyframes(keyframes: Sequence[Keyframe], frame: int) -> float:
    if not keyframes:
        return 0.0

    if frame <= keyframes[0].frame:
        return keyframes[0].value
    if frame >= keyframes[-1].frame:
        return keyframes[-1].value

    for left, right in zip(keyframes, keyframes[1:]):
        if left.frame <= frame <= right.frame:
            distance = max(right.frame - left.frame, 1)
            t = (frame - left.frame) / distance
            t = _apply_easing(t, right.easing)
            return left.value + (right.value - left.value) * t

    return keyframes[-1].value


def _resolve_track(config: Dict[str, Any], aliases: Sequence[str], default: float, last_frame: int) -> List[Keyframe]:
    animation_block = config.get("animation", {})
    tracks_block = config.get("tracks", {})
    properties_block = config.get("properties", {})
    for alias in aliases:
        if alias in config:
            return _coerce_keyframes(config[alias], last_frame, default)
        if isinstance(animation_block, dict) and alias in animation_block:
            return _coerce_keyframes(animation_block[alias], last_frame, default)
        if isinstance(tracks_block, dict) and alias in tracks_block:
            return _coerce_keyframes(tracks_block[alias], last_frame, default)
        if isinstance(properties_block, dict) and alias in properties_block:
            return _coerce_keyframes(properties_block[alias], last_frame, default)
    return _coerce_keyframes(None, last_frame, default)


def _normalize_animation_config(animation_json: str) -> Dict[str, Any]:
    text = animation_json.strip()
    if not text:
        raise ValueError("animation_json 不能为空。")

    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as handle:
            text = handle.read()

    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"animation_json 不是有效的 JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("animation_json 顶层必须是 JSON 对象。")
    return config


def _get_frame_count(config: Dict[str, Any], fps: int) -> int:
    if "frame_count" in config:
        return max(1, int(config["frame_count"]))
    if "frames" in config and isinstance(config["frames"], int):
        return max(1, int(config["frames"]))
    if "duration_seconds" in config:
        return max(1, int(round(float(config["duration_seconds"]) * fps)))
    if "duration" in config and isinstance(config["duration"], (int, float)):
        return max(1, int(round(float(config["duration"]) * fps)))
    return 24


def _bbox_or_full(alpha: Image.Image) -> Sequence[int]:
    bbox = alpha.getbbox()
    if bbox is None:
        return (0, 0, alpha.width, alpha.height)
    return bbox


def _extract_element_layer(image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
    rgba = image.convert("RGBA")
    alpha = mask.convert("L")
    rgba.putalpha(alpha)
    bbox = _bbox_or_full(alpha)
    return {
        "element": rgba.crop(bbox),
        "bbox": bbox,
    }


def _fit_base_image(base_image: Image.Image, target_size: Sequence[int]) -> Image.Image:
    converted = base_image.convert("RGBA")
    if converted.size == tuple(target_size):
        return converted
    return converted.resize(tuple(target_size), Image.Resampling.LANCZOS)


def _paste_transformed_element(
    background: Image.Image,
    element: Image.Image,
    bbox: Sequence[int],
    translate_x: float,
    translate_y: float,
    scale_x: float,
    scale_y: float,
    rotation: float,
    opacity: float,
) -> Image.Image:
    scale_x = max(0.01, scale_x)
    scale_y = max(0.01, scale_y)
    opacity = max(0.0, min(1.0, opacity))

    target_width = max(1, int(round(element.width * scale_x)))
    target_height = max(1, int(round(element.height * scale_y)))
    transformed = element.resize((target_width, target_height), Image.Resampling.BICUBIC)
    transformed = transformed.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)

    if opacity < 1.0:
        alpha = transformed.getchannel("A")
        alpha = alpha.point(lambda px: int(round(px * opacity)))
        transformed.putalpha(alpha)

    canvas = background.copy()
    center_x = (bbox[0] + bbox[2]) / 2.0 + translate_x
    center_y = (bbox[1] + bbox[3]) / 2.0 + translate_y
    paste_x = int(round(center_x - transformed.width / 2.0))
    paste_y = int(round(center_y - transformed.height / 2.0))
    canvas.alpha_composite(transformed, dest=(paste_x, paste_y))
    return canvas


def _resolve_output_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        return folder_paths.get_output_directory()
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _build_output_path(prefix: str, width: int, height: int) -> Tuple[str, str, str]:
    if folder_paths is not None and hasattr(folder_paths, "get_save_image_path"):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix,
            _resolve_output_dir(),
            width,
            height,
        )
    else:  # pragma: no cover
        full_output_folder = _resolve_output_dir()
        filename = prefix.replace("/", "_").replace("\\", "_")
        counter = 1
        subfolder = ""

    os.makedirs(full_output_folder, exist_ok=True)
    file_name = f"{filename}_{counter:05}_.mp4"
    return os.path.join(full_output_folder, file_name), file_name, subfolder


def _normalize_path_for_downstream(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def _build_video_metadata(
    full_path: str,
    file_name: str,
    subfolder: str,
    fps: int,
    frame_count: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    normalized_path = _normalize_path_for_downstream(full_path)
    return {
        "filename": file_name,
        "subfolder": subfolder,
        "type": "output",
        "format": "video/mp4",
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "absolute_path": normalized_path,
        "file_url": f"file:///{normalized_path}",
    }


def _save_video_mp4(frames: Sequence[Image.Image], fps: int, filename_prefix: str, crf: int) -> Tuple[str, str, str]:
    if not frames:
        raise ValueError("没有可编码的视频帧。")

    full_path, file_name, subfolder = _build_output_path(filename_prefix, frames[0].width, frames[0].height)

    last_error: Exception | None = None
    encoded_with_h264 = False

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is not None:
        try:
            _encode_via_ffmpeg(frames, fps, crf, full_path, ffmpeg_exe)
            encoded_with_h264 = True
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if not encoded_with_h264:
        try:
            _encode_via_av(frames, fps, crf, full_path)
        except Exception as exc:
            if last_error is not None:
                raise last_error from exc
            raise

    normalized_full_path = _normalize_path_for_downstream(full_path)
    if not os.path.exists(normalized_full_path.replace("/", os.sep)):
        raise RuntimeError(f"视频写出失败，文件不存在: {normalized_full_path}")
    file_size = os.path.getsize(normalized_full_path.replace("/", os.sep))
    if file_size <= 0:
        raise RuntimeError(f"视频写出失败，文件大小为 0: {normalized_full_path}")
    return normalized_full_path, file_name, subfolder


def _get_ffmpeg_exe() -> str | None:
    """返回可用的 ffmpeg 可执行文件路径。

    优先尝试 imageio-ffmpeg 自带的 ffmpeg（pip 安装时自动下载）。
    其次尝试系统 PATH 中的 ffmpeg。
    都不可用则返回 None，节点会回退到 av 编码。
    """
    if _imageio_ffmpeg is not None:
        try:
            return _imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    for name in ("ffmpeg", "ffmpeg.exe"):
        from shutil import which
        found = which(name)
        if found:
            return found
    return None


def _encode_via_ffmpeg(
    frames: Sequence[Image.Image],
    fps: int,
    crf: int,
    full_path: str,
    ffmpeg_exe: str,
) -> None:
    """通过 ffmpeg 子进程把帧编码为 h264 mp4。

    优点：输出是浏览器原生支持的 h264/yuv420p。
    实现：先写到一个临时 rawvideo 文件，再让 ffmpeg 从文件读，
    避免高分辨率时 stdin pipe BrokenPipe。
    """
    # h264/yuv420p 要求宽高都是偶数
    raw_width = frames[0].width
    raw_height = frames[0].height
    enc_width = raw_width + (raw_width % 2)
    enc_height = raw_height + (raw_height % 2)

    import tempfile
    raw_path = os.path.join(tempfile.gettempdir(), f"decor_anim_{uuid.uuid4().hex}.rgb")
    try:
        with open(raw_path, "wb") as handle:
            for frame in frames:
                arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
                if (arr.shape[1], arr.shape[0]) != (enc_width, enc_height):
                    # 转 PIL 再 resize 到偶数尺寸
                    pil = Image.fromarray(arr, mode="RGB")
                    if pil.size != (enc_width, enc_height):
                        pil = pil.resize((enc_width, enc_height), Image.Resampling.LANCZOS)
                    arr = np.asarray(pil, dtype=np.uint8)
                handle.write(arr.tobytes())

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{enc_width}x{enc_height}",
            "-pix_fmt", "rgb24",
            "-r", str(max(int(fps), 1)),
            "-i", raw_path,
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", str(max(0, min(51, int(crf)))),
            "-movflags", "+faststart",
            full_path,
        ]
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg 编码失败: "
                + result.stderr.decode("utf-8", errors="ignore")[-400:]
            )
    finally:
        try:
            os.remove(raw_path)
        except OSError:
            pass


def _encode_via_av(frames: Sequence[Image.Image], fps: int, crf: int, full_path: str) -> None:
    """通过 PyAV 直接编码。

    在很多环境里 libx264 二进制不能初始化，所以这里默认用 mpeg4 编码。
    这是回退方案，输出文件虽然合法但浏览器不一定能播。
    """
    container = av.open(full_path, mode="w", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("mpeg4", rate=Fraction(max(int(fps), 1), 1))
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"

        for frame in frames:
            rgb_frame = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


class DecorAnimationPlayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "base_image": ("IMAGE",),
                "animation_json": ("STRING", {"multiline": True, "default": "{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_y\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": -40, \"easing\": \"ease_in_out\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 23, \"value\": 1.15, \"easing\": \"ease_in_out\"}],\n  \"rotation\": [{\"frame\": 0, \"value\": -6}, {\"frame\": 23, \"value\": 6, \"easing\": \"ease_in_out\"}],\n  \"opacity\": 1.0\n}"}),
            },
            "optional": {
                "fps_override": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "decor_animation/decor_animation"}),
                "mp4_crf": ("INT", {"default": 20, "min": 0, "max": 51, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video",)
    FUNCTION = "animate"
    CATEGORY = "Ai说说/动画"

    def animate(self, image, mask, base_image, animation_json, fps_override=0, filename_prefix="decor_animation/decor_animation", mp4_crf=20):
        config = _normalize_animation_config(animation_json)
        fps = int(fps_override) if int(fps_override) > 0 else int(config.get("fps", 12))
        fps = max(1, fps)
        frame_count = _get_frame_count(config, fps)
        last_frame = max(frame_count - 1, 0)

        translate_x_track = _resolve_track(config, ("translate_x", "x", "position_x", "tx"), 0.0, last_frame)
        translate_y_track = _resolve_track(config, ("translate_y", "y", "position_y", "ty"), 0.0, last_frame)
        scale_track = _resolve_track(config, ("scale",), 1.0, last_frame)
        scale_x_track = _resolve_track(config, ("scale_x", "sx"), 1.0, last_frame)
        scale_y_track = _resolve_track(config, ("scale_y", "sy"), 1.0, last_frame)
        rotation_track = _resolve_track(config, ("rotation", "angle"), 0.0, last_frame)
        opacity_track = _resolve_track(config, ("opacity", "alpha"), 1.0, last_frame)

        element_image = _tensor_image_to_pil(image)
        mask_pil = _tensor_mask_to_pil(mask, element_image.size)
        base_pil = _fit_base_image(_tensor_image_to_pil(base_image), element_image.size)
        layers = _extract_element_layer(element_image, mask_pil)
        frame_images: List[Image.Image] = []
        for frame_index in range(frame_count):
            uniform_scale = _sample_keyframes(scale_track, frame_index)
            scale_x = uniform_scale * _sample_keyframes(scale_x_track, frame_index)
            scale_y = uniform_scale * _sample_keyframes(scale_y_track, frame_index)

            frame_rgba = _paste_transformed_element(
                background=base_pil,
                element=layers["element"],
                bbox=layers["bbox"],
                translate_x=_sample_keyframes(translate_x_track, frame_index),
                translate_y=_sample_keyframes(translate_y_track, frame_index),
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=_sample_keyframes(rotation_track, frame_index),
                opacity=_sample_keyframes(opacity_track, frame_index),
            )
            frame_images.append(frame_rgba)

        video_path, file_name, subfolder = _save_video_mp4(frame_images, fps, filename_prefix, int(mp4_crf))
        video_metadata = _build_video_metadata(
            full_path=video_path,
            file_name=file_name,
            subfolder=subfolder,
            fps=fps,
            frame_count=frame_count,
            width=frame_images[0].width,
            height=frame_images[0].height,
        )
        ui_payload = {
            "videos": [video_metadata],
            "text": [f"saved video: {file_name} ({frame_count} frames @ {fps} fps)"],
        }
        return {
            "ui": ui_payload,
            "result": (json.dumps(video_metadata, ensure_ascii=False),),
        }


NODE_CLASS_MAPPINGS = {
    "DecorAnimationPlayer": DecorAnimationPlayer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DecorAnimationPlayer": "Decor Animation Player",
}
