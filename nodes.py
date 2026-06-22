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

try:
    from comfy_api.latest import ComfyExtension, io, ui
except ImportError:  # pragma: no cover
    ComfyExtension = None
    io = None
    ui = None


HAS_OFFICIAL_PREVIEW_API = ComfyExtension is not None and io is not None and ui is not None


# #region debug-point A:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: Dict[str, Any] | None = None, run_id: str = "pre-fix") -> None:
    import json as _json
    import urllib.request as _request

    env_path = os.path.join(os.path.dirname(__file__), ".dbg", "video-card-preview.env")
    server_url = "http://127.0.0.1:7777/event"
    session_id = "video-card-preview"
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        server_url = next((line.split("=", 1)[1] for line in content.splitlines() if line.startswith("DEBUG_SERVER_URL=")), server_url)
        session_id = next((line.split("=", 1)[1] for line in content.splitlines() if line.startswith("DEBUG_SESSION_ID=")), session_id)
    except Exception:
        return
    try:
        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
        }
        _request.urlopen(
            _request.Request(
                server_url,
                data=_json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1.0,
        ).read()
    except Exception:
        pass
# #endregion


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


def _fit_image_to_size(image: Image.Image, target_size: Sequence[int]) -> Image.Image:
    converted = image.convert("RGB")
    if converted.size == tuple(target_size):
        return converted
    return converted.resize(tuple(target_size), Image.Resampling.LANCZOS)


def _pil_to_comfy_image(pil: Image.Image) -> torch.Tensor:
    """把单张 PIL.Image 转成 ComfyUI 标准的 IMAGE tensor: [1, H, W, 3] float32 RGB."""
    rgb = pil.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _pil_batch_to_comfy_image(pils: Sequence[Image.Image]) -> torch.Tensor:
    """把一批 PIL.Image 转成 ComfyUI 标准的 IMAGE tensor: [N, H, W, 3] float32 RGB."""
    arrays = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pils]
    if not arrays:
        return torch.zeros((0, 0, 0, 3), dtype=torch.float32)
    return torch.from_numpy(np.stack(arrays, axis=0))


def _pil_to_comfy_mask(pil: Image.Image) -> torch.Tensor:
    """把单张 PIL.Image 转成 ComfyUI 标准的 MASK tensor: [1, H, W] float32."""
    alpha = pil.convert("L")
    array = np.asarray(alpha, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.endswith("px"):
            stripped = stripped[:-2].strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_axis_value(value: Any, reference: float) -> float | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            try:
                return reference * float(stripped[:-1].strip()) / 100.0
            except ValueError:
                return None
    return _coerce_float(value)


def _get_dict_value(data: Dict[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if alias in data:
            return data[alias]
    return None


def _first_dict(data: Any, aliases: Sequence[str]) -> Dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    for alias in aliases:
        value = data.get(alias)
        if isinstance(value, dict):
            return value
    return None


def _normalize_position_config(position_json: str) -> Dict[str, Any]:
    text = position_json.strip()
    if not text:
        raise ValueError("position_json 不能为空。")

    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as handle:
            text = handle.read()

    try:
        config = json.loads(_strip_json_code_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"position_json 不是有效的 JSON: {exc}") from exc

    if isinstance(config, list):
        config = next((item for item in config if isinstance(item, dict)), None)

    if not isinstance(config, dict):
        raise ValueError("position_json 顶层必须是 JSON 对象。")
    return config


def _select_layout_block(config: Dict[str, Any]) -> Dict[str, Any]:
    current = config
    seen_ids = set()
    while isinstance(current, dict) and id(current) not in seen_ids:
        seen_ids.add(id(current))
        nested = _first_dict(
            current,
            (
                "layout",
                "placement",
                "position",
                "rect",
                "box",
                "bbox",
                "sticker",
                "result",
                "data",
            ),
        )
        if nested is None:
            return current
        current = nested
    return config


def _extract_reference_canvas_size(config: Dict[str, Any], layout: Dict[str, Any]) -> Tuple[float | None, float | None]:
    candidate_blocks = [
        _first_dict(config, ("size", "canvas", "reference_image", "source_image", "source", "base_image", "image", "meta")),
        _first_dict(layout, ("size", "canvas", "reference_image", "source_image", "source", "base_image", "image", "meta")),
        config,
    ]
    for block in candidate_blocks:
        if not isinstance(block, dict):
            continue
        width = _get_dict_value(
            block,
            (
                "canvas_width",
                "reference_width",
                "source_width",
                "image_width",
                "base_width",
                "original_width",
                "width",
            ),
        )
        height = _get_dict_value(
            block,
            (
                "canvas_height",
                "reference_height",
                "source_height",
                "image_height",
                "base_height",
                "original_height",
                "height",
            ),
        )
        width = _coerce_float(width)
        height = _coerce_float(height)
        if width and height:
            return width, height
    return None, None


def _resolve_layout_rect(
    config: Dict[str, Any],
    base_size: Sequence[int],
    element_size: Sequence[int],
) -> Dict[str, float]:
    base_width = float(base_size[0])
    base_height = float(base_size[1])
    element_width = float(element_size[0])
    element_height = float(element_size[1])
    layout = _select_layout_block(config)
    ref_width, ref_height = _extract_reference_canvas_size(config, layout)
    ref_width = ref_width or base_width
    ref_height = ref_height or base_height
    scale_to_base_x = base_width / max(ref_width, 1.0)
    scale_to_base_y = base_height / max(ref_height, 1.0)

    left = top = width = height = center_x = center_y = None

    bbox = _get_dict_value(layout, ("bbox_xywh", "box_xywh", "rect_xywh"))
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        left = _coerce_axis_value(bbox[0], ref_width)
        top = _coerce_axis_value(bbox[1], ref_height)
        width = _coerce_axis_value(bbox[2], ref_width)
        height = _coerce_axis_value(bbox[3], ref_height)

    if left is None or top is None or width is None or height is None:
        bbox = _get_dict_value(layout, ("bbox", "box", "rect"))
        if isinstance(bbox, dict):
            left = left if left is not None else _coerce_axis_value(
                _get_dict_value(bbox, ("left", "x", "x1")), ref_width
            )
            top = top if top is not None else _coerce_axis_value(
                _get_dict_value(bbox, ("top", "y", "y1")), ref_height
            )
            width = width if width is not None else _coerce_axis_value(
                _get_dict_value(bbox, ("width", "w")), ref_width
            )
            height = height if height is not None else _coerce_axis_value(
                _get_dict_value(bbox, ("height", "h")), ref_height
            )
            right = _coerce_axis_value(_get_dict_value(bbox, ("right", "x2")), ref_width)
            bottom = _coerce_axis_value(_get_dict_value(bbox, ("bottom", "y2")), ref_height)
            if width is None and left is not None and right is not None:
                width = right - left
            if height is None and top is not None and bottom is not None:
                height = bottom - top
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            raw_left = _coerce_axis_value(bbox[0], ref_width)
            raw_top = _coerce_axis_value(bbox[1], ref_height)
            raw_third = _coerce_axis_value(bbox[2], ref_width)
            raw_fourth = _coerce_axis_value(bbox[3], ref_height)
            if None not in (raw_left, raw_top, raw_third, raw_fourth):
                left = raw_left
                top = raw_top
                if raw_third > raw_left and raw_fourth > raw_top:
                    width = raw_third - raw_left
                    height = raw_fourth - raw_top
                else:
                    width = raw_third
                    height = raw_fourth

    if left is None:
        left = _coerce_axis_value(_get_dict_value(layout, ("left", "x")), ref_width)
    if top is None:
        top = _coerce_axis_value(_get_dict_value(layout, ("top", "y")), ref_height)
    if width is None:
        width = _coerce_axis_value(_get_dict_value(layout, ("width", "w")), ref_width)
    if height is None:
        height = _coerce_axis_value(_get_dict_value(layout, ("height", "h")), ref_height)

    center_x = _coerce_axis_value(_get_dict_value(layout, ("center_x", "cx")), ref_width)
    center_y = _coerce_axis_value(_get_dict_value(layout, ("center_y", "cy")), ref_height)
    right = _coerce_axis_value(_get_dict_value(layout, ("right", "x2")), ref_width)
    bottom = _coerce_axis_value(_get_dict_value(layout, ("bottom", "y2")), ref_height)

    if width is None or height is None:
        scale = _coerce_float(_get_dict_value(layout, ("scale",)))
        scale_x = _coerce_float(_get_dict_value(layout, ("scale_x", "sx")))
        scale_y = _coerce_float(_get_dict_value(layout, ("scale_y", "sy")))
        scale_x = scale_x if scale_x is not None else (scale if scale is not None else 1.0)
        scale_y = scale_y if scale_y is not None else (scale if scale is not None else 1.0)
        if width is None:
            width = element_width * max(scale_x, 0.01)
        if height is None:
            height = element_height * max(scale_y, 0.01)

    width = max((width or element_width) * scale_to_base_x, 1.0)
    height = max((height or element_height) * scale_to_base_y, 1.0)

    if left is not None:
        left *= scale_to_base_x
    if top is not None:
        top *= scale_to_base_y
    if center_x is not None:
        center_x *= scale_to_base_x
    if center_y is not None:
        center_y *= scale_to_base_y
    if right is not None:
        right *= scale_to_base_x
    if bottom is not None:
        bottom *= scale_to_base_y

    if left is None and right is not None:
        left = right - width
    if top is None and bottom is not None:
        top = bottom - height
    if center_x is None:
        center_x = (left + width / 2.0) if left is not None else base_width / 2.0
    if center_y is None:
        center_y = (top + height / 2.0) if top is not None else base_height / 2.0

    rotation = _coerce_float(_get_dict_value(layout, ("rotation", "angle"))) or 0.0
    opacity = _coerce_float(_get_dict_value(layout, ("opacity", "alpha")))
    opacity = 1.0 if opacity is None else max(0.0, min(1.0, opacity))

    return {
        "center_x": center_x,
        "center_y": center_y,
        "target_width": width,
        "target_height": height,
        "rotation": rotation,
        "opacity": opacity,
    }


def _resize_element_to_fit_box(element: Image.Image, target_width: float, target_height: float) -> Image.Image:
    # 保持贴纸有效区域的原始宽高比，完整放进 placement 指定的目标框，避免拉伸变形。
    max_width = max(1, int(round(target_width)))
    max_height = max(1, int(round(target_height)))
    scale = min(max_width / max(element.width, 1), max_height / max(element.height, 1))
    scale = max(scale, 0.01)
    resize_width = max(1, int(round(element.width * scale)))
    resize_height = max(1, int(round(element.height * scale)))
    return element.resize((resize_width, resize_height), Image.Resampling.BICUBIC)


def _apply_layout_to_element(
    base_image: Image.Image,
    element: Image.Image,
    position_json: str,
) -> Tuple[Image.Image, Image.Image]:
    config = _normalize_position_config(position_json)
    rect = _resolve_layout_rect(config, base_image.size, element.size)
    transformed = _resize_element_to_fit_box(
        element,
        rect["target_width"],
        rect["target_height"],
    )
    transformed = transformed.rotate(
        -rect["rotation"],
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )

    if rect["opacity"] < 1.0:
        alpha = transformed.getchannel("A")
        alpha = alpha.point(lambda px: int(round(px * rect["opacity"])))
        transformed.putalpha(alpha)

    paste_x = int(round(rect["center_x"] - transformed.width / 2.0))
    paste_y = int(round(rect["center_y"] - transformed.height / 2.0))

    canvas = base_image.copy()
    canvas.paste(transformed, (paste_x, paste_y), transformed)

    placed_mask = Image.new("L", base_image.size, 0)
    placed_mask.paste(transformed.getchannel("A"), (paste_x, paste_y))
    return canvas, placed_mask


def _build_positioned_sticker(
    sticker_image: torch.Tensor,
    sticker_mask: torch.Tensor,
    position_json: str,
    base_size: Sequence[int],
) -> Tuple[Image.Image, Image.Image]:
    image_pil = _tensor_image_to_pil(sticker_image)
    mask_pil = _tensor_mask_to_pil(sticker_mask, image_pil.size)
    layer = _extract_element_layer(image_pil, mask_pil)
    element = layer["element"]
    base_rgba = Image.new("RGBA", tuple(base_size), (0, 0, 0, 0))
    return _apply_layout_to_element(base_rgba, element, position_json)


def _compose_stickers(
    sticker_1: torch.Tensor,
    mask_1: torch.Tensor,
    position_json_1: str,
    sticker_2: torch.Tensor,
    mask_2: torch.Tensor,
    position_json_2: str,
    sticker_3: torch.Tensor,
    mask_3: torch.Tensor,
    position_json_3: str,
    base_image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_rgba = _tensor_image_to_pil(base_image).convert("RGBA")
    sticker_layer_1, output_mask_1 = _build_positioned_sticker(sticker_1, mask_1, position_json_1, base_rgba.size)
    sticker_layer_2, output_mask_2 = _build_positioned_sticker(sticker_2, mask_2, position_json_2, base_rgba.size)
    sticker_layer_3, output_mask_3 = _build_positioned_sticker(sticker_3, mask_3, position_json_3, base_rgba.size)

    composed = base_rgba.copy()
    composed.alpha_composite(sticker_layer_1)
    composed.alpha_composite(sticker_layer_2)
    composed.alpha_composite(sticker_layer_3)

    return (
        _pil_to_comfy_image(composed),
        _pil_to_comfy_mask(output_mask_1),
        _pil_to_comfy_mask(output_mask_2),
        _pil_to_comfy_mask(output_mask_3),
    )


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


def _threshold_mask_for_bbox(mask: Image.Image, threshold: int = 8) -> Image.Image:
    alpha = mask.convert("L")
    threshold = max(0, min(255, int(threshold)))
    return alpha.point(lambda px: 255 if px >= threshold else 0)


def _extract_element_layer(image: Image.Image, mask: Image.Image) -> Dict[str, Any]:
    rgba = image.convert("RGBA")
    alpha = mask.convert("L")
    rgba.putalpha(alpha)
    bbox = _bbox_or_full(_threshold_mask_for_bbox(alpha))
    return {
        "element": rgba.crop(bbox),
        "bbox": bbox,
    }


def _fit_base_image(base_image: Image.Image, target_size: Sequence[int]) -> Image.Image:
    converted = base_image.convert("RGBA")
    if converted.size == tuple(target_size):
        return converted
    return converted.resize(tuple(target_size), Image.Resampling.LANCZOS)


def _build_element_animation(
    image: torch.Tensor,
    mask: torch.Tensor,
    animation_json: str,
    target_size: Sequence[int],
    last_frame: int,
) -> Dict[str, Any]:
    config = _normalize_animation_config(animation_json)
    fitted_image = _fit_image_to_size(_tensor_image_to_pil(image), target_size)
    mask_pil = _tensor_mask_to_pil(mask, target_size)
    layers = _extract_element_layer(fitted_image, mask_pil)
    return {
        "config": config,
        "layer": layers,
        "translate_x_track": _resolve_track(config, ("translate_x", "x", "position_x", "tx"), 0.0, last_frame),
        "translate_y_track": _resolve_track(config, ("translate_y", "y", "position_y", "ty"), 0.0, last_frame),
        "scale_track": _resolve_track(config, ("scale",), 1.0, last_frame),
        "scale_x_track": _resolve_track(config, ("scale_x", "sx"), 1.0, last_frame),
        "scale_y_track": _resolve_track(config, ("scale_y", "sy"), 1.0, last_frame),
        "rotation_track": _resolve_track(config, ("rotation", "angle"), 0.0, last_frame),
        "opacity_track": _resolve_track(config, ("opacity", "alpha"), 1.0, last_frame),
    }


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


def _build_legacy_video_metadata(
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


def _build_video_metadata(
    full_path: str,
    file_name: str,
    subfolder: str,
    fps: int,
    frame_count: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    if HAS_OFFICIAL_PREVIEW_API:
        return ui.SavedResult(file_name, subfolder, io.FolderType.output)
    return _build_legacy_video_metadata(
        full_path=full_path,
        file_name=file_name,
        subfolder=subfolder,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _build_preview_ui(file_name: str, subfolder: str):
    if HAS_OFFICIAL_PREVIEW_API:
        return ui.PreviewVideo([ui.SavedResult(file_name, subfolder, io.FolderType.output)])
    return None


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


def _render_animation_frames(
    image_1: torch.Tensor,
    mask_1: torch.Tensor,
    animation_json_1: str,
    image_2: torch.Tensor,
    mask_2: torch.Tensor,
    animation_json_2: str,
    image_3: torch.Tensor,
    mask_3: torch.Tensor,
    animation_json_3: str,
    base_image: torch.Tensor,
    fps_override: int = 0,
) -> Tuple[List[Image.Image], int, int]:
    animation_jsons = [animation_json_1, animation_json_2, animation_json_3]
    configs = [_normalize_animation_config(text) for text in animation_jsons]
    fps = int(fps_override) if int(fps_override) > 0 else max(int(config.get("fps", 12)) for config in configs)
    fps = max(1, fps)
    frame_count = max(_get_frame_count(config, fps) for config in configs)
    last_frame = max(frame_count - 1, 0)

    base_rgb = _tensor_image_to_pil(base_image)
    base_pil = _fit_base_image(base_rgb, base_rgb.size)
    target_size = base_pil.size

    element_animations = [
        _build_element_animation(image_1, mask_1, animation_json_1, target_size, last_frame),
        _build_element_animation(image_2, mask_2, animation_json_2, target_size, last_frame),
        _build_element_animation(image_3, mask_3, animation_json_3, target_size, last_frame),
    ]

    frame_images: List[Image.Image] = []
    for frame_index in range(frame_count):
        frame_rgba = base_pil.copy()
        for element_animation in element_animations:
            uniform_scale = _sample_keyframes(element_animation["scale_track"], frame_index)
            scale_x = uniform_scale * _sample_keyframes(element_animation["scale_x_track"], frame_index)
            scale_y = uniform_scale * _sample_keyframes(element_animation["scale_y_track"], frame_index)
            frame_rgba = _paste_transformed_element(
                background=frame_rgba,
                element=element_animation["layer"]["element"],
                bbox=element_animation["layer"]["bbox"],
                translate_x=_sample_keyframes(element_animation["translate_x_track"], frame_index),
                translate_y=_sample_keyframes(element_animation["translate_y_track"], frame_index),
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=_sample_keyframes(element_animation["rotation_track"], frame_index),
                opacity=_sample_keyframes(element_animation["opacity_track"], frame_index),
            )
        frame_images.append(frame_rgba)

    return frame_images, fps, frame_count


def _tensor_frames_to_pil(frames: torch.Tensor) -> Tuple[List[Image.Image], int, int, int]:
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames 必须是 IMAGE tensor, 实际收到 {type(frames).__name__}")
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"frames 形状必须是 [N, H, W, 3/4], 实际是 {tuple(frames.shape)}")
    if frames.shape[0] <= 0:
        raise ValueError("frames 是空批次")

    frame_count = int(frames.shape[0])
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    array = frames.detach().cpu().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    array = (array * 255.0).round().astype(np.uint8)
    pil_frames = [Image.fromarray(array[i], mode="RGB") for i in range(frame_count)]
    return pil_frames, frame_count, width, height


class DecorAnimationPlayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "mask_1": ("MASK",),
                "animation_json_1": ("STRING", {"multiline": True, "default": "{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_y\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": -40, \"easing\": \"ease_in_out\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 23, \"value\": 1.15, \"easing\": \"ease_in_out\"}],\n  \"rotation\": [{\"frame\": 0, \"value\": -6}, {\"frame\": 23, \"value\": 6, \"easing\": \"ease_in_out\"}],\n  \"opacity\": 1.0\n}"}),
                "image_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "animation_json_2": ("STRING", {"multiline": True, "default": "{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_x\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": 32, \"easing\": \"ease_in_out\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 23, \"value\": 0.92, \"easing\": \"ease_in_out\"}],\n  \"rotation\": 0,\n  \"opacity\": 1.0\n}"}),
                "image_3": ("IMAGE",),
                "mask_3": ("MASK",),
                "animation_json_3": ("STRING", {"multiline": True, "default": "{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_y\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 12, \"value\": 18, \"easing\": \"ease_out\"}, {\"frame\": 23, \"value\": 0, \"easing\": \"ease_in\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 12, \"value\": 1.08, \"easing\": \"ease_in_out\"}, {\"frame\": 23, \"value\": 1.0, \"easing\": \"ease_in_out\"}],\n  \"rotation\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": -10, \"easing\": \"ease_in_out\"}],\n  \"opacity\": 1.0\n}"}),
                "base_image": ("IMAGE",),
            },
            "optional": {
                "fps_override": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "decor_animation/decor_animation"}),
                "mp4_crf": ("INT", {"default": 20, "min": 0, "max": 51, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "animate"
    CATEGORY = "Ai说说/动画"

    def animate(
        self,
        image_1,
        mask_1,
        animation_json_1,
        image_2,
        mask_2,
        animation_json_2,
        image_3,
        mask_3,
        animation_json_3,
        base_image,
        fps_override=0,
        filename_prefix="decor_animation/decor_animation",
        mp4_crf=20,
    ):
        frame_images, fps, frame_count = _render_animation_frames(
            image_1=image_1,
            mask_1=mask_1,
            animation_json_1=animation_json_1,
            image_2=image_2,
            mask_2=mask_2,
            animation_json_2=animation_json_2,
            image_3=image_3,
            mask_3=mask_3,
            animation_json_3=animation_json_3,
            base_image=base_image,
            fps_override=fps_override,
        )
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
        frames_tensor = _pil_batch_to_comfy_image(frame_images)
        ui_payload = {"videos": [video_metadata], "text": [f"saved video: {file_name} ({frame_count} frames @ {fps} fps)"]}
        # #region debug-point B:player-return
        _debug_report("B", "DecorAnimationPlayer.animate:return", "player return ui.videos", {
            "ui_keys": list(ui_payload.keys()),
            "videos_count": len(ui_payload.get("videos", [])),
            "video_metadata": video_metadata,
            "file_exists": os.path.exists(video_path.replace("/", os.sep)),
            "file_size": os.path.getsize(video_path.replace("/", os.sep)) if os.path.exists(video_path.replace("/", os.sep)) else -1,
            "result_len": 1,
            "image_shape": list(frames_tensor.shape),
        })
        # #endregion
        return {
            "ui": ui_payload,
            "result": (frames_tensor,),
        }


class DecorStickerLayoutComposer:
    @classmethod
    def INPUT_TYPES(cls):
        default_position_json = (
            '{\n'
            '  "id": "1",\n'
            '  "size": {\n'
            '    "width": 912,\n'
            '    "height": 1145\n'
            '  },\n'
            '  "placement": {\n'
            '    "left": 56,\n'
            '    "top": 72,\n'
            '    "width": 240,\n'
            '    "height": 240,\n'
            '    "rotation": -8,\n'
            '    "opacity": 1.0\n'
            '  }\n'
            '}'
        )
        return {
            "required": {
                "sticker_1": ("IMAGE",),
                "mask_1": ("MASK",),
                "position_json_1": ("STRING", {"multiline": True, "default": default_position_json}),
                "sticker_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "position_json_2": ("STRING", {"multiline": True, "default": default_position_json}),
                "sticker_3": ("IMAGE",),
                "mask_3": ("MASK",),
                "position_json_3": ("STRING", {"multiline": True, "default": default_position_json}),
                "base_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask_1", "mask_2", "mask_3")
    FUNCTION = "compose"
    CATEGORY = "Ai说说/排版"

    def compose(
        self,
        sticker_1,
        mask_1,
        position_json_1,
        sticker_2,
        mask_2,
        position_json_2,
        sticker_3,
        mask_3,
        position_json_3,
        base_image,
    ):
        return _compose_stickers(
            sticker_1=sticker_1,
            mask_1=mask_1,
            position_json_1=position_json_1,
            sticker_2=sticker_2,
            mask_2=mask_2,
            position_json_2=position_json_2,
            sticker_3=sticker_3,
            mask_3=mask_3,
            position_json_3=position_json_3,
            base_image=base_image,
        )


class DecorImageSizeText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    CATEGORY = "Ai说说/工具"

    def get_text(self, image):
        pil = _tensor_image_to_pil(image)
        text = f"宽度：{pil.width}\n高度：{pil.height}"
        return (text,)


class DecorFrameSequencePreview:
    """把上游输出的 IMAGE 帧序列直接在前端播放成动画。

    接 `Decor Animation Player` 的 `image` 端口，把 `IMAGE` 帧序列编码成
    浏览器原生可播的 h264 mp4，并通过 `ui.videos` 在节点卡片里播放。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 12, "min": 1, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "decor_animation/decor_sequence"}),
            },
            "optional": {
                "mp4_crf": ("INT", {"default": 20, "min": 0, "max": 51, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    CATEGORY = "Ai说说/动画"
    OUTPUT_NODE = True

    def preview(
        self,
        frames: torch.Tensor,
        fps: int = 12,
        filename_prefix: str = "decor_animation/decor_sequence",
        mp4_crf: int = 20,
    ):
        pil_frames, frame_count, width, height = _tensor_frames_to_pil(frames)
        fps = max(1, int(fps))
        full_path, file_name, subfolder = _save_video_mp4(
            pil_frames, fps, filename_prefix, mp4_crf
        )

        video_metadata = _build_video_metadata(
            full_path=full_path,
            file_name=file_name,
            subfolder=subfolder,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )

        ui_payload = {
            "videos": [video_metadata],
            "text": [f"preview video: {file_name} ({frame_count} frames @ {fps} fps)"],
        }
        # #region debug-point C:preview-return
        _debug_report("C", "DecorFrameSequencePreview.preview:return", "preview return ui.videos", {
            "ui_keys": list(ui_payload.keys()),
            "videos_count": len(ui_payload.get("videos", [])),
            "video_metadata": video_metadata,
            "file_exists": os.path.exists(full_path.replace("/", os.sep)),
            "file_size": os.path.getsize(full_path.replace("/", os.sep)) if os.path.exists(full_path.replace("/", os.sep)) else -1,
            "result_len": 1,
            "frames_shape": list(frames.shape),
        })
        # #endregion
        return {
            "ui": ui_payload,
            "result": (frames,),
        }


if HAS_OFFICIAL_PREVIEW_API:
    class DecorAnimationPlayerLatest(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DecorAnimationPlayer",
                display_name="Decor Animation Player",
                category="Ai说说/动画",
                inputs=[
                    io.Image.Input("image_1"),
                    io.Mask.Input("mask_1"),
                    io.String.Input(
                        "animation_json_1",
                        multiline=True,
                        default="{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_y\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": -40, \"easing\": \"ease_in_out\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 23, \"value\": 1.15, \"easing\": \"ease_in_out\"}],\n  \"rotation\": [{\"frame\": 0, \"value\": -6}, {\"frame\": 23, \"value\": 6, \"easing\": \"ease_in_out\"}],\n  \"opacity\": 1.0\n}",
                    ),
                    io.Image.Input("image_2"),
                    io.Mask.Input("mask_2"),
                    io.String.Input(
                        "animation_json_2",
                        multiline=True,
                        default="{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_x\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": 32, \"easing\": \"ease_in_out\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 23, \"value\": 0.92, \"easing\": \"ease_in_out\"}],\n  \"rotation\": 0,\n  \"opacity\": 1.0\n}",
                    ),
                    io.Image.Input("image_3"),
                    io.Mask.Input("mask_3"),
                    io.String.Input(
                        "animation_json_3",
                        multiline=True,
                        default="{\n  \"fps\": 12,\n  \"frame_count\": 24,\n  \"translate_y\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 12, \"value\": 18, \"easing\": \"ease_out\"}, {\"frame\": 23, \"value\": 0, \"easing\": \"ease_in\"}],\n  \"scale\": [{\"frame\": 0, \"value\": 1.0}, {\"frame\": 12, \"value\": 1.08, \"easing\": \"ease_in_out\"}, {\"frame\": 23, \"value\": 1.0, \"easing\": \"ease_in_out\"}],\n  \"rotation\": [{\"frame\": 0, \"value\": 0}, {\"frame\": 23, \"value\": -10, \"easing\": \"ease_in_out\"}],\n  \"opacity\": 1.0\n}",
                    ),
                    io.Image.Input("base_image"),
                    io.Int.Input("fps_override", default=0, min=0, max=120, step=1),
                    io.String.Input("filename_prefix", default="decor_animation/decor_animation"),
                    io.Int.Input("mp4_crf", default=20, min=0, max=51, step=1),
                ],
                outputs=[io.Image.Output(display_name="image")],
                is_output_node=True,
            )

        @classmethod
        def execute(
            cls,
            image_1,
            mask_1,
            animation_json_1,
            image_2,
            mask_2,
            animation_json_2,
            image_3,
            mask_3,
            animation_json_3,
            base_image,
            fps_override=0,
            filename_prefix="decor_animation/decor_animation",
            mp4_crf=20,
        ):
            frame_images, fps, _frame_count = _render_animation_frames(
                image_1=image_1,
                mask_1=mask_1,
                animation_json_1=animation_json_1,
                image_2=image_2,
                mask_2=mask_2,
                animation_json_2=animation_json_2,
                image_3=image_3,
                mask_3=mask_3,
                animation_json_3=animation_json_3,
                base_image=base_image,
                fps_override=fps_override,
            )
            video_path, file_name, subfolder = _save_video_mp4(frame_images, fps, filename_prefix, int(mp4_crf))
            _ = video_path
            frames_tensor = _pil_batch_to_comfy_image(frame_images)
            return io.NodeOutput(
                frames_tensor,
                ui=_build_preview_ui(file_name, subfolder),
            )


    class DecorStickerLayoutComposerLatest(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            default_position_json = (
                '{\n'
                '  "id": "1",\n'
                '  "size": {\n'
                '    "width": 912,\n'
                '    "height": 1145\n'
                '  },\n'
                '  "placement": {\n'
                '    "left": 56,\n'
                '    "top": 72,\n'
                '    "width": 240,\n'
                '    "height": 240,\n'
                '    "rotation": -8,\n'
                '    "opacity": 1.0\n'
                '  }\n'
                '}'
            )
            return io.Schema(
                node_id="DecorStickerLayoutComposer",
                display_name="Decor Sticker Layout Composer",
                category="Ai说说/排版",
                inputs=[
                    io.Image.Input("sticker_1"),
                    io.Mask.Input("mask_1"),
                    io.String.Input("position_json_1", multiline=True, default=default_position_json),
                    io.Image.Input("sticker_2"),
                    io.Mask.Input("mask_2"),
                    io.String.Input("position_json_2", multiline=True, default=default_position_json),
                    io.Image.Input("sticker_3"),
                    io.Mask.Input("mask_3"),
                    io.String.Input("position_json_3", multiline=True, default=default_position_json),
                    io.Image.Input("base_image"),
                ],
                outputs=[
                    io.Image.Output(display_name="image"),
                    io.Mask.Output(display_name="mask_1"),
                    io.Mask.Output(display_name="mask_2"),
                    io.Mask.Output(display_name="mask_3"),
                ],
            )

        @classmethod
        def execute(
            cls,
            sticker_1,
            mask_1,
            position_json_1,
            sticker_2,
            mask_2,
            position_json_2,
            sticker_3,
            mask_3,
            position_json_3,
            base_image,
        ):
            return io.NodeOutput(
                *_compose_stickers(
                    sticker_1=sticker_1,
                    mask_1=mask_1,
                    position_json_1=position_json_1,
                    sticker_2=sticker_2,
                    mask_2=mask_2,
                    position_json_2=position_json_2,
                    sticker_3=sticker_3,
                    mask_3=mask_3,
                    position_json_3=position_json_3,
                    base_image=base_image,
                )
            )


    class DecorImageSizeTextLatest(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DecorImageSizeText",
                display_name="Decor Image Size Text",
                category="Ai说说/工具",
                inputs=[
                    io.Image.Input("image"),
                ],
                outputs=[io.String.Output(display_name="text")],
            )

        @classmethod
        def execute(cls, image):
            pil = _tensor_image_to_pil(image)
            return io.NodeOutput(f"宽度：{pil.width}\n高度：{pil.height}")


    class DecorFrameSequencePreviewLatest(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DecorFrameSequencePreview",
                display_name="Decor Frame Sequence Preview",
                category="Ai说说/动画",
                inputs=[
                    io.Image.Input("frames"),
                    io.Int.Input("fps", default=12, min=1, max=120, step=1),
                    io.String.Input("filename_prefix", default="decor_animation/decor_sequence"),
                    io.Int.Input("mp4_crf", default=20, min=0, max=51, step=1),
                ],
                outputs=[io.Image.Output(display_name="image")],
                is_output_node=True,
            )

        @classmethod
        def execute(
            cls,
            frames,
            fps=12,
            filename_prefix="decor_animation/decor_sequence",
            mp4_crf=20,
        ):
            pil_frames, _frame_count, _width, _height = _tensor_frames_to_pil(frames)
            full_path, file_name, subfolder = _save_video_mp4(
                pil_frames,
                max(1, int(fps)),
                filename_prefix,
                int(mp4_crf),
            )
            _ = full_path
            return io.NodeOutput(
                frames,
                ui=_build_preview_ui(file_name, subfolder),
            )


    class DecorAnimationExtension(ComfyExtension):
        async def get_node_list(self):
            return [
                DecorAnimationPlayerLatest,
                DecorStickerLayoutComposerLatest,
                DecorImageSizeTextLatest,
                DecorFrameSequencePreviewLatest,
            ]


    async def comfy_entrypoint() -> DecorAnimationExtension:
        return DecorAnimationExtension()


NODE_CLASS_MAPPINGS = {
    "DecorAnimationPlayer": DecorAnimationPlayer,
    "DecorStickerLayoutComposer": DecorStickerLayoutComposer,
    "DecorImageSizeText": DecorImageSizeText,
    "DecorFrameSequencePreview": DecorFrameSequencePreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DecorAnimationPlayer": "Decor Animation Player",
    "DecorStickerLayoutComposer": "Decor Sticker Layout Composer",
    "DecorImageSizeText": "Decor Image Size Text",
    "DecorFrameSequencePreview": "Decor Frame Sequence Preview",
}
