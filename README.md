# ComfyUI Decor Animation Player

一个可直接放进 `ComfyUI/custom_nodes` 的自定义节点集合，用来把 3 组装饰元素图片分别按各自的动画 JSON 表达式播放、叠加到底图上，并直接输出可在 `ComfyUI` 前端播放的 `MP4` 视频。同时也提供一个静态排版节点，可根据 VLM 输出的位置 JSON 把 3 个贴纸摆放到底图上。

## 节点列表

- `Decor Animation Player` —— 读取 3 组装饰元素 + mask + 动画 JSON，再叠加到底图上，合成每帧并输出 `IMAGE` 序列，同时直接生成并预览视频
- `Decor Sticker Layout Composer` —— 读取 3 组贴纸 + mask + 位置 JSON，把贴纸按排版信息缩放、旋转并叠加到底图上，同时输出 3 个排版后的 mask
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
5. 在节点分类 `Ai说说/排版` 中找到静态贴纸排版节点

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

## 节点 3：Decor Sticker Layout Composer

把 3 个贴纸根据各自的位置 JSON 排版到底图上，适合接收 VLM 生成的排版结果。

### 输入

- `sticker_1`: 第 1 个贴纸图片
- `mask_1`: 第 1 个贴纸对应的遮罩
- `position_json_1`: 第 1 个贴纸的位置 JSON
- `sticker_2`: 第 2 个贴纸图片
- `mask_2`: 第 2 个贴纸对应的遮罩
- `position_json_2`: 第 2 个贴纸的位置 JSON
- `sticker_3`: 第 3 个贴纸图片
- `mask_3`: 第 3 个贴纸对应的遮罩
- `position_json_3`: 第 3 个贴纸的位置 JSON
- `base_image`: 被贴纸覆盖的底图

### 输出

- `image`: 三个贴纸排版叠加后的最终图像
- `mask_1`: 贴纸 1 经缩放、排版后回投到底图坐标系的 mask
- `mask_2`: 贴纸 2 经缩放、排版后回投到底图坐标系的 mask
- `mask_3`: 贴纸 3 经缩放、排版后回投到底图坐标系的 mask
- 节点会先按 `mask` 做阈值裁切，再取贴纸有效区域参与排版，避免弱水印或底噪把裁切区域错误放大
- 静态排版节点只控制贴纸放在底图中的位置和缩放，不改变传入贴纸本身的旋转和透明度
- `width` / `height` / `scale` / `scale_x` / `scale_y` 可用于缩放贴纸，节点会保持原始比例，不拉伸变形
- JSON 中即使传入 `rotation`、`opacity` 也会被忽略

### 位置 JSON

支持以下常见字段，字段名可以混用：

- 位置：`x` / `y` / `left` / `top`
- 中心点：`center_x` / `center_y` / `cx` / `cy`
- 尺寸：`width` / `height` / `w` / `h`
- 边界框：`bbox` / `box` / `rect`
- 缩放：`scale` / `scale_x` / `scale_y`
- 参考画布尺寸：`canvas_width` / `canvas_height` / `source_width` / `source_height`

最简单的示例：

```json
{
  "canvas_width": 1080,
  "canvas_height": 1920,
  "x": 120,
  "y": 240,
  "width": 320,
  "height": 320
}
```

也支持 `bbox` 写法：

```json
{
  "canvas_width": 1080,
  "canvas_height": 1920,
  "bbox": {
    "left": 120,
    "top": 240,
    "width": 320,
    "height": 320
  }
}
```

### 推荐给 VLM 的固定格式

虽然节点兼容多种别名写法，但如果你要让 `VLM` 稳定输出、并直接喂给这个节点，建议固定成下面这个单贴纸格式，不要混用别名：

```json
{
  "id": "1",
  "size": {
    "width": 912,
    "height": 1145
  },
  "placement": {
    "left": 56,
    "top": 72,
    "width": 240,
    "height": 240
  }
}
```

推荐原因：

- 只保留一套字段名，避免 `x/y`、`bbox`、`center_x` 混用
- `canvas.width` 和 `canvas.height` 明确说明这是 VLM 参考原图尺寸
- `placement.left/top/width/height` 语义清晰，节点可以直接解析
- 所有值都用像素和数值，避免百分比、中文单位和模糊描述

### VLM 输出约束

建议你在提示词里明确要求模型遵守这些规则：

- 只输出纯 JSON，不要输出 Markdown，不要包在代码块里
- 顶层必须是对象，不要输出数组
- 必须包含 `id`、`size`、`placement`
- `size.width`、`size.height` 必须等于 VLM 观察时原图尺寸
- `placement.left`、`top`、`width`、`height` 必须是像素数值
- 不要输出解释文字、置信度、推理过程、建议语句
- 不要输出百分比字符串、`px` 字符串、自然语言位置描述

### 三贴纸总响应模板

如果你想让 VLM 一次性返回三张贴纸的位置，推荐总响应格式固定如下，然后把 `stickers` 里的三个对象分别接到节点的 `position_json_1/2/3`：

```json
{
  "size": {
    "width": 912,
    "height": 1145
  },
  "stickers": [
    {
      "id": "1",
      "placement": {
        "left": 56,
        "top": 72,
        "width": 240,
        "height": 240
      }
    },
    {
      "id": "2",
      "placement": {
        "left": 626,
        "top": 118,
        "width": 220,
        "height": 220
      }
    },
    {
      "id": "3",
      "placement": {
        "left": 286,
        "top": 1290,
        "width": 340,
        "height": 220
      }
    }
  ]
}
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
