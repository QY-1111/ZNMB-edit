# ComfyUI Decor Animation Player

一个可直接放进 `ComfyUI/custom_nodes` 的自定义节点，用来把装饰元素图片按照动画 JSON 表达式播放成动画帧，并在节点执行后生成可预览的动图。

## 功能

- 输入一张图片
- 输入该图片对应的 `mask`
- 输入动画 `JSON` 文本，或直接传入一个 `.json` 文件路径
- 输出动画帧批次 `IMAGE`
- 输出对应透明度批次 `MASK`
- 生成节点内可预览的 GIF 动图

## 安装

1. 把整个仓库目录放到 `ComfyUI/custom_nodes/` 下
2. 重启 `ComfyUI`
3. 在节点分类 `Ai说说/动画` 中找到 `Decor Animation Player`

## 节点输入

- `image`: 装饰元素图片
- `mask`: 装饰元素遮罩
- `animation_json`: 动画配置，支持直接粘贴 JSON，也支持填写本地 `.json` 文件路径
- `background_mode`:
  - `transparent_canvas`: 输出透明背景上的动画
  - `keep_unmasked`: 保留遮罩外的原图背景
- `fps_override`: 大于 `0` 时覆盖 JSON 里的 `fps`

## 节点输出

- `frames`: `IMAGE` 批次，每一帧是动画结果
- `alpha_masks`: `MASK` 批次，每一帧对应透明度
- `preview_file`: 生成的预览 GIF 文件路径

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

- 建议输入图片本身就是要动的装饰元素，`mask` 用来精确限制参与动画的区域
- 节点会根据 `mask` 自动裁出元素区域，再做位移、缩放、旋转和透明度变化
- 预览图会写到 `ComfyUI/temp/decor_animation_previews/`

## 示例

示例 JSON 见 `examples/sample_animation.json`
