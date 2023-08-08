---
sidebar_position: 5
---

# 注意事项

## 相比社区 QAT 接口使用异同

### 算子融合函数

|                                                  | torch            | Plugin                                                 |
| ------------------------------------------------ | ---------------- | ------------------------------------------------------ |
| horizon_plugin_pytorch.quantization.fuse_modules | 调用该接口       | 调用该接口                                             |
| 上述接口的 fuser_func 参数设置                   | 使用社区内部接口 | horizon_plugin_pytorch.quantization.fuse_known_modules |

如上表所示，社区和Plugin在融合算子时，均需调用 `horizon_plugin_pytorch.quantization.fuse_modules` 接口：

```python
fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=_fuse_known_modules, fuse_custom_config_dict=None)
```

不同点是使用 Plugin 进行量化训练，需将 `horizon_plugin_pytorch.quantization.fuse_known_modules` 接口作为参数对上述接口中的 `fuser_func` 进行赋值来使用 Plugin 所定义的算子融合规则。

### QConfig 参数设置

|                              | torch        | Plugin                                                       |
| ---------------------------- | ------------ | ------------------------------------------------------------ |
| torch.quantization.QConfig   | 调用该接口   | 调用该接口                                                   |
| QConfig 成员变量：activation | 社区提供参数 | horizon_plugin_pytorch.quantization.default_8bit_fake_quant  |
| QConfig 成员变量：weight     | 社区提供参数 | horizon_plugin_pytorch.quantization.default_weight_8bit_fake_quant |

如上表所示，在设置模型的 `qconfig` 时，社区和 Plugin 均使用 `torch.qconfig.QConfig` 。但是 Plugin 自定义了 `QConfig` 在初始化时需要的对输出或是权值进行量化的参数。

Plugin 也提供了两个接口用于获取常用的 `QConfig` ：

```python
horizon_plugin_pytorch.quantization.get_default_qat_qconfig(bits=8, backend="")
horizon_plugin_pytorch.quantization.get_default_qat_out_qconfig(bits=8, backend="")
```

用户可以直接通过上述接口获取相应的 `QConfig` 。

### 不同阶段间模型转换

Plugin 提供了 `horizon_plugin_pytorch.quantization.prepare_qat` 实现浮点模型向 QAT 模型的转换，提供了 `horizon_plugin_pytorch.quantization.convert` 实现 QAT 模型向定点预测阶段模型转换。