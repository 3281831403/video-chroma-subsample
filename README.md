# video-chroma-subsample

## 题面

把 4:4:4 素材转成 4:2:0 后，边缘出现锯齿、色度还偏移半像素：① 4:2:0 下采样直接抽样而不是 2×2 平均，斜边出现锯齿；② 奇数宽高补边用 0 而不是复制边缘，右边缘出现黑条；③ 上采样用最近邻，色块明显；④ 上采样宽高各差 1，输出尺寸不准；⑤ 矩阵固定 BT.601，BT.709 素材偏色；⑥ limited/full 不区分，暗部被抬升；⑦ 色度上采样不按样本中心对齐，整体偏移半个像素；⑧ 平面尺寸不匹配不报错，越界读取；⑨ 批量转换共用平面缓冲，帧与帧之间串数据；⑩ 未知 matrix 静默当 BT.601。chroma/subsample.py 是核心，tests/test_subsample.py 15个红。要修的是：4:2:0 的 2×2 平均与边缘复制补边、4:2:2 的水平 2:1 抽样、4:2:0/4:2:2 的双线性上采样与半像素中心对齐、尺寸精确、BT.601/BT.709 正反矩阵、limited/full 范围、平面尺寸校验与 ChromaError、批量原子回滚与帧间缓冲隔离、未知 matrix 报错。

## 运行方式

```powershell
cd D:\gsb-20261119\video-chroma-subsample
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `chroma/subsample.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_subsample.py` 是评测基准，**不可修改**；只允许修改 `chroma/subsample.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数。
