---
sidebar_position: 6
---

# 数据校准

## Calibration (Experimental Support){#calibration}

在 plugin 的量化训练中，一个重要的步骤是确定量化参数 `scale` ，一个合理的 `scale` 能够显著提升模型训练结果和加快模型的收敛速度。
一种常见的 `scale` 的计算方法为：

```py
def compute_scale(data, quant_min, quant_max):
    fmax = data.abs().max()
    scale = fmax * 2 / (quant_max - quant_min)
    return scale
```

当计算 feature map 的 `scale` 时，由于每次 forward 只能计算出当前 batch 的 `fmax`，对于整个数据集来说，每次 forward 计算出来的 feature map 可能不准确。因此，引入了 `calibration` 方法。

### Calibration 方法

Calibration 方法是在量化训练之前，使用浮点模型统计计算 `scale` 的方法。步骤为：

1. 浮点模型 forward，collect 浮点模型的统计数据。

2. 使用步骤 1 的统计数据，通过 `Calibration` 得到 feature map 的量化参数。

3. 使用步骤 2 得到的量化参数，初始化量化训练模型的量化参数。

4. 在步骤 3 的基础上进行量化训练。

### Plugin Calibration 使用方法

plugin 提供了默认的 Calibration 配置，用户可以通过设置 `float.qconfig = get_default_calib_qconfig()` 来使用 calibration 功能。

```py
horizon.quantization.get_default_calib_qconfig()
```

### plugin Calibration 的限制

1. 只支持对 feature map 做 Calibration 。
2. 不支持 `train()` 模式和 `eval()` 模式行为不一致的 Module 。

## Calibration v2(Experimental Support)

Horizon Plugin Pytorch 于 1.2.1 版本后支持了新的 `calibration` 用法，与原有 calibration 相比，新的 calibration 支持更多的 calibration 方法，用法更灵活，推荐您优先尝试新版 calibration 用法。原有 calibration 用法依然兼容，但在之后的版本中会逐渐弃用。

### 使用流程

calibration 与 QAT 的整体流程如下图所示：

![calibration_v2_workflow](./image/horizon_expert/calibration_v2_workflow.svg)

下面分别介绍各个步骤：

1. 构建并训练浮点模型。参考 plugin快速上手章节中的 [构建浮点模型](#build-float-model) 和 [浮点模型预训练](#float-model-pretrain) 小节内容。

2. 将浮点模型转化 QAT 模型。参考 plugin快速上手章节中的 [设置BPU架构](#set-bpu) 、 [算子融合](#op-fuse) 和 [浮点模型转为量化模型](#float-to-quantized) 小节。使用 `prepare_qat` 方法转化浮点模型前，需要为模型设置 `qconfig` 。
   
    ```python
    model.qconfig = horizon.quantization.get_default_qconfig()
    ```

    `get_default_qconfig` 可以为 `weight` 和 `activation` 设置不同的 `fake_quant` 和 `observer` 。目前，支持的 `fake quant` 方法有 "fake_quant"、"lsq" 和 "pact"，支持的 `observer` 有 "min_max"、 "fixed_scale"、"clip"、"percentile" 和 "clip_std"。如无特殊需求， `activation_fake_quant` 和 `weight_fake_quant` 推荐使用默认的 "fake_quant" 方法， `weight_observer` 使用默认的 "min_max"。如果为 QAT 阶段设置 qconfig ， `activation_observer` 推荐使用默认的 "min_max"，如果为 calibration 阶段设置 qconfig ， `activation_observer` 推荐使用 "percentile"。 calibration 可选 `observer` 有 "min_max"、 "percentile" 和 "clip_std", 特殊用法和调试技巧见 calibration 经验总结。

    ```python
    def get_default_qconfig(
        activation_fake_quant: Optional[str] = "fake_quant",
        weight_fake_quant: Optional[str] = "fake_quant",
        activation_observer: Optional[str] = "min_max",
        weight_observer: Optional[str] = "min_max",
        activation_qkwargs: Optional[Dict] = None,
        weight_qkwargs: Optional[Dict] = None,
    ):
    ```

3. 设置 `fake quantize` 状态为 `CALIBRATION` 。

    ```python
    horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.CALIBRATION)
    ```

    `fake quantize` 一共有三种状态，分别需要在 `QAT` 、 `calibration` 、 `validation` 前将模型的 `fake quantize` 设置为对应的状态。在 calibration 状态下，仅观测各算子输入输出的统计量。在 QAT 状态下，除观测统计量外还会进行伪量化操作。而在 validation 状态下，不会观测统计量，仅进行伪量化操作。

    ```python
    class FakeQuantState(Enum):
        QAT = "qat"
        CALIBRATION = "calibration"
        VALIDATION = "validation"
    ```

4. calibration 。把准备好的校准数据喂给模型，模型在 forward 过程中由 observer 观测相关统计量。

5. 设置 `fake quantize` 状态为 `VALIDATION` 。

    ```python
    horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.VALIDATION)
    ```

6. 验证 `calibration` 效果。如果效果满意，则进入步骤 7 ，不满意则调整 `calibration qconfig` 中的参数继续 calibration 。

7. 从浮点模型开始重新按照步骤 2 的流程构建 QAT 模型，需要注意 `qconfig` 设置与 calibration 阶段的区别。

8. 加载 calibration 得到的参数。

    ```python
    horizon.quantization.load_observer_params(calibration_model, qat_model)
    ```

9.  设置 `fake quantize` 状态为 `QAT` 。

    ```python
    horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.QAT)
    ```

10. QAT 训练。

11. 设置 `fake quantize` 状态为 `VALIDATION` ，并验证 QAT 模型精度。

    ```python
    horizon.quantization.set_fake_quantize(model, horizon.quantization.FakeQuantState.VALIDATION)
    ```

### 使用限制

不支持 `train()` 模式和 `eval()` 模式行为不一致的Module。