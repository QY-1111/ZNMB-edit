# ComfyUI Decor Animation Player

一个可直接放进 `ComfyUI/custom_nodes` 的自定义节点，用来把装饰元素图片按照动画 JSON 表达式播放，并叠加到底图上，最终直接输出一个 `MP4` 视频。

## 功能

- 输入一张图片
- 输入该图片对应的 `mask`
- 输入动画 `JSON` 文本，或直接传入一个 `.json` 文件路径
- 输入一个底图，作为最终画面的固定底板
- 直接输出最终 `MP4` 视频
- 节点内可预览视频结果

## 安装

1. 把整个仓库目录放到 `ComfyUI/custom_nodes/` 下
2. 重启 `ComfyUI`
3. 在节点分类 `Ai说说/动画` 中找到 `Decor Animation Player`

## 节点输入

- `image`: 要做动画的装饰元素图片
- `mask`: 装饰元素遮罩
- `base_image`: 最终画面的底图
- `animation_json`: 动画配置，支持直接粘贴 JSON，也支持填写本地 `.json` 文件路径
- `fps_override`: 大于 `0` 时覆盖 JSON 里的 `fps`
- `filename_prefix`: 输出视频文件名前缀
- `mp4_crf`: MP4 编码质量，越小画质越高、体积越大

## 节点输出

- `video`: 生成的视频文件路径

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

- `image` 建议只放需要动的装饰元素，`mask` 用来精确限制参与动画的区域
- 节点会根据 `mask` 自动裁出元素区域，再做位移、缩放、旋转和透明度变化
- 每一帧会把动画元素合成到 `base_image` 上，最后编码成 `MP4`
- 输出视频会写到 `ComfyUI/output/`

## 示例

示例 JSON 见 `examples/sample_animation.json`
