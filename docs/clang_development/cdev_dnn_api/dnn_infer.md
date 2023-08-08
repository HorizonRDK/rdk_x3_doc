---
sidebar_position: 4
---

# 模型推理 API


## hbDNNInfer()


**【函数原型】**  

``int32_t hbDNNInfer(hbDNNTaskHandle_t *taskHandle, hbDNNTensor **output, const hbDNNTensor *input, hbDNNHandle_t dnnHandle, hbDNNInferCtrlParam *inferCtrlParam)``

**【功能描述】** 

根据输入参数执行推理任务。调用方可以跨函数、跨线程使用返回的 ``taskHandle``。

**【参数】**

- [out]     ``taskHandle``          任务句柄指针。
- [in/out]  ``output``              推理任务的输出。
- [in]      ``input``               推理任务的输入。
- [in]      ``dnnHandle``           DNN句柄指针。
- [in]      ``inferCtrlParam``      控制推理任务的参数。

**【返回类型】** 

- 返回 ``0`` 则表示API成功执行，否则执行失败。

:::info 备注

  使用该接口提交任务时应提前将 ``taskHandle`` 置为 ``nullptr``，除非是给指定 ``taskHandle`` 追加任务（即使用 ``inferCtrlParam::more`` 功能）。

  最多支持同时存在32个模型任务。

  对于batch模型，允许分开设置输入张量的内存地址。例如：模型的输入validShape/alignedShape为[4, 3, 224, 224], 可以申请四个hbDNNTensor， 每个hbDNNTensor的validShape/alignedShape都设置为[1, 3, 224, 224],存放每个batch的数据。当模型有多个输入时， ``input`` 的顺序应为input0[batch0], input0[batch1], ..., inputn[batch0], inputn[batch1], ...。
:::

## hbDNNRoiInfer()


**【函数原型】**  

``int32_t hbDNNRoiInfer(hbDNNTaskHandle_t *taskHandle, hbDNNTensor **output, const hbDNNTensor *input, hbDNNRoi *rois, int32_t roiCount, hbDNNHandle_t dnnHandle, hbDNNInferCtrlParam *inferCtrlParam)``

**【功能描述】** 

根据输入参数执行ROI推理任务。根据输入参数执行ROI推理任务。调用方可以跨函数、跨线程使用返回的 ``taskHandle``。

**【参数】**

- [out]     ``taskHandle``       任务句柄指针。
- [in/out]  ``output``           推理任务的输出。
- [in]      ``input``            推理任务的输入。
- [in]      ``rois``             Roi框信息。
- [in]      ``roiCount``         Roi框数量。
- [in]      ``dnnHandle``        dnn句柄指针。
- [in]      ``inferCtrlParam``   控制推理任务的参数。

**【返回类型】** 

- 返回 ``0`` 则表示API成功执行，否则执行失败。

:::info 备注

  | 该接口支持批处理操作，假设需要推理的数据批数为 ``batch``，模型输入个数为 ``input_count``，其中resizer输入源的数量为 ``resizer_count``。
  | 准备输入参数 ``input``：第i个 ``batch`` 对应的 ``input`` 数组下标范围是 :math:`[i * input\_count`, :math:`(i + 1) * input\_count)，i=[0,batch)`;
  | 准备输入参数 ``rois``：每个resizer输入源的输入都应匹配一个roi，第i个 ``batch`` 对应的 ``rois`` 数组下标范围是 :math:`[i * resizer\_count`, :math:`(i + 1) * resizer\_count)，i=[0,batch)`; 每个batch的roi顺序应和输入的顺序保持一致；
  | 关于 ``batch`` 数量限制：其范围应该在[1, 255];

  模型限制：模型需要在编译时将编译参数 ``input_source`` 设置为 ``resizer``, 模型的 h*w 要小于18432;

  使用该接口提交任务时应提前将 ``taskHandle`` 置为 ``nullptr``，除非是给指定 ``taskHandle`` 追加任务（即使用 ``inferCtrlParam::more`` 功能）。

  最多支持同时存在32个模型任务。

  API示例： 可参考[模型推理DNN API使用示例说明文档](/toolchain_development/intermediate/)的 ``roi_infer.sh`` 说明。

  模型限制：在模型转换时，将编译参数 input_source 设置为 {'input_name': 'resizer'}即可生成resizer模型，具体参数配置细节可参考[PTQ量化原理及步骤说明的转换模型](/toolchain_development/intermediate/ptq_process#model_conversion)中的介绍。

  ![resizer](./image/cdev_dnn_api/resizer.png)

  目前也支持多输入的nv12数据，resizer常用的输出尺寸(HxW)：128x128、128x64、64x128、160x96
:::

## hbDNNWaitTaskDone()


**【函数原型】**  

``int32_t hbDNNWaitTaskDone(hbDNNTaskHandle_t taskHandle, int32_t timeout)``

**【功能描述】** 

等待任务完成或超时。

**【参数】**

- [in]  ``taskHandle``         任务句柄指针。
- [in]  ``timeout``            超时配置（单位：毫秒）。

**【返回类型】** 

- 返回 ``0`` 则表示API成功执行，否则执行失败。

:::info 备注

  1. ``timeout > 0`` 表示等待时间；
  2. ``timeout <= 0`` 表示一直等待，直到任务完成。
:::

## hbDNNReleaseTask()


**【函数原型】**  

``int32_t hbDNNReleaseTask(hbDNNTaskHandle_t taskHandle)``

**【功能描述】** 

释放任务，如果任务未执行则会直接取消并释放任务，如果已经执行则会在运行到某些节点后取消并释放任务。

**【参数】**

- [in]  ``taskHandle``         任务句柄指针。

**【返回类型】** 

- 返回 ``0`` 则表示API成功执行，否则执行失败。
