# ComfyUI Decor Animation Player

一个可直接放进 `ComfyUI/custom_nodes` 的自定义节点集合，用来把 3 组装饰元素图片分别按各自的动画 JSON 表达式播放、叠加到底图上，并直接输出可在 `ComfyUI` 前端播放的 `MP4` 视频。

## 节点列表

- `Decor Animation Player` —— 读取 3 组装饰元素 + mask + 动画 JSON，再叠加到底图上，合成每帧并输出 `IMAGE` 序列，同时直接生成并预览视频
- `Decor Frame Sequence Preview` —— 把上游 `IMAGE` 帧序列直接编码成可在 `ComfyUI` 前端播放的视频

## 功能

- 输入三张装饰元素图片
- 输入三张装饰元素对应的 `mask`
- 输入三段动画 `JSON` 文本，或直接传入三个 `.json` 文件路径
- 输入一个底图，作为最终画面的固定底板
- 直接输出最终 `MP4` 视频
- 节点会通过 `ComfyUI` 原生 `ui.videos` 在前端显示视频预览

## 安装

1. 把整个仓库目录放到 `ComfyUI/custom_nodes/` 下
2. `pip install -r requirements.txt` 安装 `imageio-ffmpeg` 等依赖
3. 重启 `ComfyUI`
4. 在节点分类 `Ai说说/动画` 中找到这两个节点

> 节点会优先调用 `imageio-ffmpeg` 自带的 `ffmpeg` 子进程写出 **h264 / yuv420p** 的 MP4，
> 这种格式浏览器原生支持，节点卡片可以直接预览。
> 如果 `imageio-ffmpeg` 不可用，会回退到 `PyAV` 的 mpeg4 编码。

## 节点 1：Decor Animation Player

### 输入

- `image_1`: 第 1 个装饰元素图片
- `mask_1`: 第 1 个装饰元素遮罩
- `animation_json_1`: 第 1 个装饰元素动画配置
- `image_2`: 第 2 个装饰元素图片
- `mask_2`: 第 2 个装饰元素遮罩
- `animation_json_2`: 第 2 个装饰元素动画配置
- `image_3`: 第 3 个装饰元素图片
- `mask_3`: 第 3 个装饰元素遮罩
- `animation_json_3`: 第 3 个装饰元素动画配置
- `base_image`: 最终画面的底图
- `fps_override`: 大于 `0` 时覆盖 JSON 里的 `fps`
- `filename_prefix`: 输出视频文件名前缀
- `mp4_crf`: MP4 编码质量，越小画质越高、体积越大

### 输出

- `image`: 所有帧合成图，`IMAGE` tensor，格式 `[N, H, W, 3]`，可直接接到 `Decor Frame Sequence Preview`、`Preview Image`、`Save Image`、`Image Batch` 等节点
- 前端预览：节点会直接返回 `ui.videos`，可在 `ComfyUI` 节点卡片中原生播放
- 视频生成：运行节点时会自动写出 `MP4`

## 节点 2：Decor Frame Sequence Preview

把上游 `IMAGE` 帧序列直接编码成可在前端播放的动画。

### 输入

- `frames`: `IMAGE` tensor，形状 `[N, H, W, 3/4]`，通常接 `Decor Animation Player` 的 `image` 端口
- `fps`: 播放帧率
- `filename_prefix`: 输出视频文件名前缀
- `mp4_crf`: MP4 编码质量

### 输出

- `image`: 透传 `frames`
- 前端预览：节点会直接返回 `ui.videos`，可在 `ComfyUI` 节点卡片中原生播放
- 视频生成：运行节点时会自动写出 `MP4`

### 推荐链路

```text
Decor Animation Player (image 输出)
        ↓ frames
Decor Frame Sequence Preview (节点卡片直接播放)
```

## JSON 格式

最简单的写法：

```json
{
  "fps": 12,
  "frame_count": 24,
  "translate_y": [
    { "frame": 0, "value": 0 },
    { "frame": 23, "value": -40, "easing": "ease_in_out" }
  ],
  "scale": [
    { "frame": 0, "value": 1.0 },
    { "frame": 23, "value": 1.15, "easing": "ease_in_out" }
  ],
  "rotation": [
    { "frame": 0, "value": -6 },
    { "frame": 23, "value": 6, "easing": "ease_in_out" }
  ],
  "opacity": 1.0
}
```

## 支持的动画属性

- `translate_x` / `x` / `position_x` / `tx`
- `translate_y` / `y` / `position_y` / `ty`
- `scale`
- `scale_x` / `sx`
- `scale_y` / `sy`
- `rotation` / `angle`
- `opacity` / `alpha`

## 关键帧写法

支持以下几种形式：

```json
{
  "rotation": 15
}
```

```json
{
  "scale": {
    "from": 1.0,
    "to": 1.2,
    "easing": "ease_in_out"
  }
}
```

```json
{
  "translate_x": [
    { "frame": 0, "value": 0 },
    { "frame": 12, "value": 30, "easing": "ease_out" },
    { "frame": 23, "value": -10, "easing": "ease_in" }
  ]
}
```

## 支持的缓动

- `linear`
- `ease_in`
- `ease_out`
- `ease_in_out`
- `sin`

## 说明

- `image_1` 到 `image_3` 建议只放需要动的装饰元素，`mask_1` 到 `mask_3` 用来精确限制参与动画的区域
- 节点会分别根据每组 `mask` 自动裁出元素区域，再按对应的 `animation_json_x` 做位移、缩放、旋转和透明度变化
- 每一帧会把三个动画元素依次合成到 `base_image` 上，最后编码成 `MP4`
- 三段 JSON 可以写完全不同的动画表达式；如果 `fps_override=0`，节点会取三段 JSON 里最大的 `fps` 作为最终输出帧率
- 输出视频会写到 `ComfyUI/output/`

## 示例

示例 JSON 见 `examples/sample_animation.json`
