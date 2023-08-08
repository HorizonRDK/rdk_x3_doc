---
sidebar_position: 4
---

# API手册

## 量化API

Fuse modules.

```python
horizon_plugin_pytorch.quantization.fuse_modules.fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=<function fuse_known_modules>, fuse_custom_config_dict=None)
```
Fuses a list of modules into a single module.

Fuses only the following sequence of modules: conv, bn; conv, bn, relu; conv, relu; conv, bn, add; conv, bn, add, relu; conv, add; conv, add, relu; linear, bn; linear, bn, relu; linear, relu; linear, bn, add; linear, bn, add, relu; linear, add; linear, add, relu. For these sequences, the first element in the output module list performs the fused operation. The rest of the elements are set to nn.Identity()

**参数**
- **model** – Model containing the modules to be fused

- **modules_to_fuse** – list of list of module names to fuse. Can also be a list of strings if there is only a single list of modules to fuse.

- **inplace** – bool specifying if fusion happens in place on the model, by default a new model is returned

- **fuser_func** – Function that takes in a list of modules and outputs a list of fused modules of the same length. For example, fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()] Defaults to torch.ao.quantization.fuse_known_modules

- **fuse_custom_config_dict** – custom configuration for fusion

```python
    # Example of fuse_custom_config_dict
    fuse_custom_config_dict = {
        # Additional fuser_method mapping
        "additional_fuser_method_mapping": {
            (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
        },
    }
```

**返回**

&emsp;model with fused modules. A new copy is created if inplace=True.


Examples:

```python
    >>> # xdoctest: +SKIP
    >>> m = M().eval()
    >>> # m is a module containing the sub-modules below
    >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'],
                        ['submodule.conv', 'submodule.relu']]
    >>> fused_m = fuse_modules(
                    m, modules_to_fuse)
    >>> output = fused_m(input)

    >>> m = M().eval()
    >>> # Alternately provide a single list of modules to fuse
    >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
    >>> fused_m = fuse_modules(
                    m, modules_to_fuse)
    >>> output = fused_m(input)
```

Prepare and convert.

```python
horizon_plugin_pytorch.quantization.quantize.convert(module, mapping=None, inplace=False, remove_qconfig=True)
```
Convert modules.

Convert submodules in input module to a different module according to mapping by calling from_float method on the target module class. And remove qconfig at the end if remove_qconfig is set to True.

**参数**

- **module** – input module

- **mapping** – a dictionary that maps from source module type to target module type, can be overwritten to allow swapping user defined Modules

- **inplace** – carry out model transformations in-place, the original module is mutated


```python
horizon_plugin_pytorch.quantization.quantize.prepare_calibration(model, inplace=False)
```
Prepare the model for calibration.

**参数**

- **model** – Float model with fused ops

- **inplace** – carry out model transformations in-place or not. Defaults to False.

```python
horizon_plugin_pytorch.quantization.quantize.prepare_qat(model: torch.nn.modules.module.Module, mapping: Optional[Dict[Type[torch.nn.modules.module.Module], Type[torch.nn.modules.module.Module]]] = None, inplace: bool = False, optimize_graph: bool = False, hybrid: bool = False, optimize_kwargs: Optional[Dict[str, Tuple]] = None)
```
Prepare qat.

Prepare a copy of the model for quantization-aware training and converts it to quantized version.

Quantization configuration should be assigned preemptively to individual submodules in .qconfig attribute.

**参数**

- **model** – input model to be modified in-place

- **mapping** – dictionary that maps float modules to quantized modules to be replaced.

- **inplace** – carry out model transformations in-place, the original module is mutated

- **optimize_graph** – whether to do some process on origin model for special purpose. Currently only support using torch.fx to fix cat input scale(only used on Bernoulli)

- **hybrid** – whether to generate a hybrid model that some intermediate operation is computed in float. There are some constraints for this functionality now: 1. The hybrid model cannot pass check_model and cannot be compiled. 2. Some quantized operation cannot directly accept input from float operation, user need to manually insert QuantStub.

- **optimize_kwargs** – a dict for optimize graph with the following format:

    ```python
    optimize_kwargs = {
        # optional, specify which type of optimization to do. Only
        # support "unify_inputs_scale" now
        "opt_types": ("unify_inputs_scale",),
    
        # optional, modules start with qualified name to optimize
        "module_prefixes": ("backbone.conv",),
    
        # optional, modules in these types will be optimize
        "module_types": (horizon.nn.qat.conv2d,),
    
        # optional, functions to optimize
        "functions": (torch.clamp,),
    
        # optional, methods to optimize. Only support
        # FloatFunctional methods now
        "methods": ("add",),
    }
    ```

```python
horizon_plugin_pytorch.quantization.quantize_fx.convert_fx(graph_module: torch.fx.graph_module.GraphModule, inplace: bool = False, convert_custom_config_dict: Optional[Dict[str, Any]] = None, _remove_qconfig: bool = True) -> horizon_plugin_pytorch.quantization.fx.graph_module.QuantizedGraphModule
```
Convert a calibrated or trained model to a quantized model.

**参数**

- **graph_module**: A prepared and calibrated/trained model (GraphModule)
- **inplace**: Carry out model transformations in-place, the original module is mutated
- **convert_custom_config_dict**: Dictionary for custom configurations for convert Function

    ```python
    convert_custom_config_dict = {
        # We automativally preserve all attributes, this option is
        # just in case and not likely to be used.
        "preserved_attributes": ["preserved_attr"],
    }
    ```
- **_remove_qconfig**: Option to remove the qconfig attributes in the model after convert. for internal use only.

**返回**

&emsp;A quantized model (GraphModule)

Example: convert fx example:

```python
    # prepared_model: the model after prepare_fx/prepare_qat_fx and
    # calibration/training
    quantized_model = convert_fx(prepared_model)
```

```python
horizon_plugin_pytorch.quantization.quantize_fx.fuse_fx(model: torch.nn.modules.module.Module, fuse_custom_config_dict: Optional[Dict[str, Any]] = None) -> horizon_plugin_pytorch.quantization.fx.graph_module.GraphModuleWithAttr
```
Fuse modules like conv+add+bn+relu etc.

Fusion rules are defined in horizon_plugin_pytorch.quantization.fx.fusion_pattern.py.

**参数**

- **model**: A torch.nn.Module model.

- **fuse_custom_config_dict**: Dictionary for custom configurations for fuse_fx, e.g.

Example: fuse_fx example:

```python 
    from torch.quantization import fuse_fx
    m = fuse_fx(m)
```


```python
horizon_plugin_pytorch.quantization.quantize_fx.prepare_calibration_fx(model, qconfig_dict: Optional[Dict[str, Any]] = None, prepare_custom_config_dict: Optional[Dict[str, Any]] = None, optimize_graph: bool = False, hybrid: bool = False, hybrid_dict: Optional[Dict[str, List]] = None) -> horizon_plugin_pytorch.quantization.fx.graph_module.ObservedGraphModule
```
Prepare the model for calibration.

Args: Same as prepare_qat_fx

```python
horizon_plugin_pytorch.quantization.quantize_fx.prepare_qat_fx(model: Union[torch.nn.modules.module.Module, torch.fx.graph_module.GraphModule], qconfig_dict: Optional[Dict[str, Any]] = None, prepare_custom_config_dict: Optional[Dict[str, Any]] = None, optimize_graph: bool = False, hybrid: bool = False, hybrid_dict: Optional[Dict[str, List]] = None) -> horizon_plugin_pytorch.quantization.fx.graph_module.ObservedGraphModule
```
Prepare a model for quantization aware training.

**参数**

- **model**：torch.nn.Module model or GraphModule model (maybe from fuse_fx)
- **qconfig_dict**: qconfig_dict is a dictionary with the following configurations.

    ```python
    qconfig_dict = {
        # optional, global config
        "": qconfig,

        # optional, used for module types
        "module_type": [
            (torch.nn.Conv2d, qconfig),
            ...,
        ],

        # optional, used for module names
        "module_name": [
            ("foo.bar", qconfig)
            ...,
        ],
        # priority (in increasing order):
        #   global, module_type, module_name, module.qconfig
        # qconfig == None means quantization should be
        # skipped for anything matching the rule.
        # The qconfig of function or method is the same as the
        # qconfig of its parent module, if it needs to be set
        # separately, please wrap this function as a module.
    }
    ```

- **prepare_custom_config_dict**: customization configuration dictionary for quantization tool:

    ```python
    prepare_custom_config_dict = {
        # We automativally preserve all attributes, this option is
        # just in case and not likely to be used.
        "preserved_attributes": ["preserved_attr"],
    }
    ```
- **optimize_graph**: whether to do some process on origin model for special purpose. Currently only support using torch.fx to fix cat input scale(only used on Bernoulli)
- **hybrid**:  Whether prepare model in hybrid mode. Default is False and model runs on BPU completely. It should be True if the model is quantized by model convert or contains some CPU ops. In hybrid mode, ops which aren’t supported by BPU and ops which are specified by the user will run on CPU. How to set qconfig: Qconfig in hybrid mode is the same as qconfig in non-hybrid mode. For BPU op, we should ensure the input of this op is quantized, the activation qconfig of its previous non-quantstub op should not be None even if its previous non-quantstub op is a CPU op. How to specify CPU op: Define CPU module_name or module_type in hybrid_dict.
- **hybrid_dict**: Hybrid dict is a dictionary to define user-specified CPU op.

    ```python
    hybrid_dict = {
        # optional, used for module types
        "module_type": [torch.nn.Conv2d, ...],
    
        # optional, used for module names
        "module_name": ["foo.bar", ...],
    }
    # priority (in increasing order): module_type, module_name
    # To set a function or method as CPU op, wrap it as a module.
    ```

**返回**

&emsp;A GraphModule with fake quant modules (configured by qconfig_dict), ready for quantization aware training.

Examples: prepare_qat_fx example:

```python
import torch
from horizon_plugin_pytorch.quantization import get_default_qat_qconfig
from horizon_plugin_pytorch.quantization import prepare_qat_fx

qconfig = get_default_qat_qconfig()
def train_loop(model, train_data):
    model.train()
    for image, target in data_loader:
        ...

qconfig_dict = {"": qconfig}
prepared_model = prepare_qat_fx(float_model, qconfig_dict)
# Run QAT training
train_loop(prepared_model, train_loop)
```
Extended tracer and wrap of torch.fx.

This file defines a inherit tracer of torch.fx.Tracer and a extended wrap to allow wrapping of user-defined Module or method, which help users do some optimization of their own module by torch.fx

```python
horizon_plugin_pytorch.utils.fx_helper.wrap(obj)
```

Extend torch.fx.warp.

This function can be:

&emsp;1) called or used as a decorator on a string to register a builtin function as a “leaf function”

&emsp;2) called or used as a decorator on a function to register this function as a “leaf function”

&emsp;3) called or used as a decorator on subclass of torch.nn.Module to register this module as a “leaf module”, and register all user defined method in this class as “leaf method”

&emsp;4) called or used as a decorator on a class method to register it as “leaf method”


same as torch.quantization.FakeQuantize.

```python
class horizon_plugin_pytorch.quantization.fake_quantize.FakeQuantize(observer: type = <class 'horizon_plugin_pytorch.quantization.observer.MovingAverageMinMaxObserver'>, saturate: bool = None, in_place: bool = False, compat_mask: bool = True, channel_len: int = 1, **observer_kwargs)
```
Simulate the quantize and dequantize operations in training time.

The output of this module is given by

x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale # noqa

- scale: 
    defines the scale factor used for quantization.

- zero_point:
    specifies the quantized value to which 0 in floating point maps to

- quant_min:
    specifies the minimum allowable quantized value.

- quant_max:
    specifies the maximum allowable quantized value.

- fake_quant_enabled:
    controls the application of fake quantization on tensors, note that statistics can still be updated.

- observer_enabled:
    controls statistics collection on tensors

- dtype:
    specifies the quantized dtype that is being emulated with fake-quantization, the allowable values is qint8 and qint16. The values of quant_min and quant_max should be chosen to be consistent with the dtype

**参数**

- **observer** – Module for observing statistics on input tensors and calculating scale and zero-point.

- **saturate** – Whether zero out the grad for value out of quanti range.

- **in_place** – Whether use in place fake quantize.

- **compat_mask** – Whether pack the bool mask into bitfield when saturate = True.

- **channel_len** – Size of data at channel dim.

- **observer_kwargs** – Arguments for the observer module

**observer**

User provided module that collects statistics on the input tensor and provides a method to calculate scale and zero-point.

**extra_repr()**

Set the extra representation of the module

To print customized extra information, you should re-implement this method in your own modules. Both single-line and multi-line strings are acceptable.

**forward(x)**

Defines the computation performed at every call.

Should be overridden by all subclasses.

:::info 小技巧
Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.
:::
**set_qparams(scale: Union[torch.Tensor, Sequence, float], zero_point: Optional[Union[torch.Tensor, Sequence, int]] = None)**

Set qparams, default symmetric.

**classmethod with_args(\*\*kwargs)**

Wrapper that allows creation of class factories.

This can be useful when there is a need to create classes with the same constructor arguments, but different instances. Can be used in conjunction with _callable_args

Example:

```python
    >>> # xdoctest: +SKIP("Undefined vars")
    >>> Foo.with_args = classmethod(_with_args)
    >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
    >>> foo_instance1 = foo_builder()
    >>> foo_instance2 = foo_builder()
    >>> id(foo_instance1) == id(foo_instance2)
    False
```

```python
class horizon_plugin_pytorch.quantization.observer.MovingAverageMinMaxObserver(averaging_constant=0.01, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, quant_min=None, quant_max=None, is_sync_quantize=False, factory_kwargs=None)
```
Refine this docstring in the future.

Observer module for computing the quantization parameters based on the moving average of the min and max values.

This observer computes the quantization parameters based on the moving averages of minimums and maximums of the incoming tensors. The module records the average minimum and maximum of incoming tensors, and uses this statistic to compute the quantization parameters.

**参数**

- **averaging_constant** – Averaging constant for min/max.

- **dtype** – Quantized data type

- **qscheme** – Quantization scheme to be used, only support per_tensor_symmetric scheme

- **reduce_range** – Reduces the range of the quantized data type by 1 bit

- **quant_min** – Minimum quantization value.

- **quant_max** – Maximum quantization value.

- **is_sync_quantize** – Whether use sync quantize

- **factory_kwargs** – Arguments for register data buffer

**forward(x_orig)**

Record the running minimum and maximum of x.

```python
class horizon_plugin_pytorch.quantization.observer.MovingAveragePerChannelMinMaxObserver(averaging_constant=0.01, ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, quant_min=None, quant_max=None, is_sync_quantize=False, factory_kwargs=None)
```
Refine this docstring in the future.

Observer module for computing the quantization parameters based on the running per channel min and max values.

This observer uses the tensor min/max statistics to compute the per channel quantization parameters. The module records the running minimum and maximum of incoming tensors, and uses this statistic to compute the quantization parameters.

**参数**

- **averaging_constant** – Averaging constant for min/max.

- **ch_axis** – Channel axis

- **dtype** – Quantized data type

- **qscheme** – Quantization scheme to be used, Only support per_channel_symmetric

- **quant_min** – Minimum quantization value.

- **quant_max** – Maximum quantization value.

- **is_sync_quantize** – whether use sync quantize

- **factory_kwargs** – Arguments for register data buffer

**forward(x_orig)**

Defines the computation performed at every call.

Should be overridden by all subclasses.

:::info 小技巧
Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.
:::

Fuse modules.

```python
horizon_plugin_pytorch.quantization.fuse_modules.fuse_known_modules(mod_list, is_qat=False, additional_fuser_method_mapping=None)
```
Fuse modules.

Return a list of modules that fuses the operations specified in the input module list.

Fuses only the following sequence of modules: conv, bn; conv, bn, relu; conv, relu; conv, bn, add; conv, bn, add, relu; conv, add; conv, add, relu; linear, bn; linear, bn, relu; linear, relu; linear, bn, add; linear, bn, add, relu; linear, add; linear, add, relu. For these sequences, the first element in the output module list performs the fused operation. The rest of the elements are set to nn.Identity()

```python
class horizon_plugin_pytorch.march.March
```
BPU platform.

``BAYES``: Bayes platform（J5处理器使用）

``BERNOULLI2``: Bernoulli2 platform（X3处理器使用）


```python
horizon_plugin_pytorch.quantization.qconfig.get_default_calib_qconfig(dtype='qint8', calib_qkwargs=None, backend='')
```
Get default calibration qconfig.

**参数**

- **dtype** (str) – quantization type, the allowable value is qint8 and qint16

- **calib_qkwargs** (dict) – A dict that contains args of CalibFakeQuantize and args of calibration observer.

- **backend** (str) – backend implementation

```python
horizon_plugin_pytorch.quantization.qconfig.get_default_qat_out_qconfig(dtype='qint8', weight_fake_quant='fake_quant', weight_qkwargs=None, backend='')
```
Get default qat out qconfig.

**参数**

- **dtype** (str) – quantization type, the allowable value is qint8 and qint16

- **weight_fake_quant** (str) – FakeQuantize type of weight, default is fake_quant.Avaliable items is fake_quant, lsq and pact

- **weight_qkwargs** (dict) – A dict contain weight Observer type, args of weight FakeQuantize and args of weight Observer.

- **backend** (str) – backend implementation

```python
horizon_plugin_pytorch.quantization.qconfig.get_default_qat_qconfig(dtype='qint8', weight_dtype='qint8', activation_fake_quant='fake_quant', weight_fake_quant='fake_quant', activation_qkwargs=None, weight_qkwargs=None, backend='')
```
Get default qat qconfig.

**参数**

- **dtype** (str) – Activation quantization type, the allowable values is qint8 and qint16

- **weight_dtype** (str) – Weight quantization type, the allowable values is qint8 and qint16

- **activation_fake_quant** (str) – FakeQuantize type of activation, default is fake_quant. Avaliable items is fake_quant, lsq, pact

- **weight_fake_quant** (str) – FakeQuantize type of weight, default is fake_quant.Avaliable items is fake_quant, lsq and pact

- **activation_qkwargs** (dict) – A dict contain activation Observer type, args of activation FakeQuantize and args of activation Observer.

- **weight_qkwargs** (dict) – A dict contain weight Observer type, args of weight FakeQuantize and args of weight Observer.

- **backend** (str) – backend implementation


```python
horizon_plugin_pytorch.utils.onnx_helper.export_to_onnx(model, args, f, export_params=True, verbose=False, training=<TrainingMode.EVAL: 0>, input_names=None, output_names=None, operator_export_type=<OperatorExportTypes.ONNX_FALLTHROUGH: 3>, opset_version=11, do_constant_folding=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None)
```
Export a (float or qat)model into ONNX format.

**参数**

- **model** (torch.nn.Module/torch.jit.ScriptModule/ScriptFunction) – the model to be exported.

- **args** (tuple or torch.Tensor) – args can be structured either as:

    a. ONLY A TUPLE OF ARGUMENTS:

    ```python
        args = (x, y, z)
    ```

    The tuple should contain model inputs such that model(*args) is a valid invocation of the model. Any non-Tensor arguments will be hard-coded into the exported model; any Tensor arguments will become inputs of the exported model, in the order they occur in the tuple.

    b. A TENSOR:

    ```python
        args = torch.Tensor([1])
    ```

    This is equivalent to a 1-ary tuple of that Tensor.

    c. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS:

    ```python
        args = (x,
                {'y': input_y,
                'z': input_z})
    ```

    All but the last element of the tuple will be passed as non-keyword arguments, and named arguments will be set from the last element. If a named argument is not present in the dictionary , it is assigned the default value, or None if a default value is not provided.

- **f** – a file-like object or a string containing a file name. A binary protocol buffer will be written to this file.

- **export_params** (bool, default True) – if True, all parameters will be exported.

- **verbose** (bool, default False) – if True, prints a description of the model being exported to stdout, doc_string will be added to graph. doc_string may contaion mapping of module scope to node name in future torch onnx.

- **training** (enum, default TrainingMode.EVAL) – if model.training is False and in training mode if model.training is True.

    - ``TrainingMode.EVAL``: export the model in inference mode.

    - ``TrainingMode.PRESERVE``: export the model in inference mode

    - ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations which might interfere with training.

- **input_names** (list of str, default empty list) – names to assign to the input nodes of the graph, in order.

- **output_names** (list of str, default empty list) – names to assign to the output nodes of the graph, in order.

- **operator_export_type** (enum, default ONNX_FALLTHROUGH) –

    - ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops (in the default opset domain).

    - ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops to standard ONNX ops in the default opset domain.

    - ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace “aten”) are exported as ATen ops.

    - ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op (in the TorchScript namespace “aten”) as a regular ONNX op. If we are unable to do so,fall back to exporting an ATen op.

- **opset_version** (int, default 11) – by default we export the model to the opset version of the onnx submodule.

- **do_constant_folding** (bool, default False) – Apply the constant-folding optimization. Constant-folding will replace some of the ops that have all constant inputs with pre-computed constant nodes.

- **dynamic_axes** (dict<str, list(int)/dict<int, str>>, default empty dict) – By default the exported model will have the shapes of all input and output tensors set to exactly match those given in args (and example_outputs when that arg is required). To specify axes of tensors as dynamic (i.e. known only at run-time), set dynamic_axes to a dict with schema:

    - ``KEY`` (str): an input or output name. Each name must also be provided in input_names or output_names.

    - ``VALUE`` (dict or list): If a dict, keys are axis indices and values are axis names. If a list, each element is an axis index.

- **keep_initializers_as_inputs** (bool, default None) – If True, all the initializers (typically corresponding to parameters) in the exported graph will also be added as inputs to the graph. If False, then initializers are not added as inputs to the graph, and only the non-parameter inputs are added as inputs. This may allow for better optimizations (e.g. constant folding) by backends/runtimes.

- **custom_opsets** (dict<str, int>, default empty dict) – A dict with schema:

    - ``KEY`` (str): opset domain name

    - ``VALUE`` (int): opset version

    If a custom opset is referenced by model but not mentioned in this dictionary, the opset version is set to 1.

Use ‘QuantWrapper’, ‘QuantStub’, ‘DeQuantStub’.

```python
horizon_plugin_pytorch.quantization.check_model(module: Union[torch.jit._script.ScriptModule, torch.nn.modules.module.Module], example_inputs: tuple, march: Optional[str] = None, input_source: Union[Sequence[str], str] = 'ddr', advice: Optional[int] = None)
```
Check if nn.Module or jit.ScriptModule can be compiled by HBDK.

Dump advices for improving performance on BPU.

**参数**

- **module** (nn.Module or jit.ScriptModule.) –

- **example_inputs** (A tuple of example inputs, in torch.tensor format.) – For jit.trace and shape inference.

- **march** (Specify the target march of bpu.) – Valid options are bayes（J5处理器使用） and bernoulli2（X3处理器使用）. If not provided, use horizon plugin global march.

- **input_source** (Specify input features' sources(ddr/resizer/pyramid)) –

- **advice** (Print HBDK compiler advices for improving the utilization of the) – model on bpu if layers of the model become slow by more than the specified time (in microseconds)

**返回**

- **flag** – 0 if pass, otherwise not.

**返回类型**

int

```python
horizon_plugin_pytorch.quantization.compile_model(module: Union[torch.jit._script.ScriptModule, torch.nn.modules.module.Module], example_inputs: tuple, hbm: str, march: Optional[str] = None, name: Optional[str] = None, input_source: Union[Sequence[str], str] = 'ddr', input_layout: Optional[str] = None, output_layout: str = 'NCHW', opt: Union[str, int] = 'O2', balance_factor: int = 2, progressbar: bool = True, jobs: int = 16, debug: bool = False, extra_args: Optional[list] = None)
```
Compile the nn.Module or jit.ScriptModule.

**参数**

- **module** (nn.Module or jit.ScriptModule.) –

- **example_inputs** (A tuple of example inputs, in torch.tensor format.) – For jit.trace and shape inference.

- **hbm** (Specify the output path of hbdk-cc.) –

- **march** (Specify the target march of bpu.) – Valid options are bayes（J5处理器使用） and bernoulli2（X3处理器使用）. If not provided, use horizon plugin global march.

- **name** (Name of the model, recorded in hbm.) – Can be obtained by hbdk-disas or hbrtGetModelNamesInHBM in runtime.

- **input_source** (Specify input features' sources(ddr/resizer/pyramid)) –

- **input_layout** (Specify input layout of all model inputs.) – Available layouts are NHWC, NCHW, BPU_RAW.

- **output_layout** (Specify input layout of all model inputs.) – Available layouts are NHWC, NCHW, BPU_RAW.

- **opt** (Specify optimization options.) – Available options are O0, O1, O2, O3, ddr, fast, balance.

- **balance_factor** (Specify the balance ratio when optimization options is) – ‘balance’.

- **progressbar** (Show compilation progress to alleviate anxiety.) –

- **jobs** (Specify number of threads launched during compiler optimization.) – Default is ‘16’. 0 means use all available hardware concurrency.

- **debug** (Enable debugging info in hbm.) –

- **extra_args** (specify extra args listed in "hbdk-cc -h".) – format in list of string: e.g. [‘–ability-entry’, str(entry_value), …]

**返回**

- **flag** – 0 if pass, otherwise not.

**返回类型**

int

```python
horizon_plugin_pytorch.quantization.export_hbir(module: Union[torch.jit._script.ScriptModule, torch.nn.modules.module.Module], example_inputs: tuple, hbir: str, march: Optional[str] = None)
```
Export the nn.Module or jit.ScriptModule to hbdk3.HBIR.

**参数**

- **module** (nn.Module or jit.ScriptModule.) –

- **example_inputs** (A tuple of example inputs, in torch.tensor format.) – For jit.trace and shape inference.

- **hbir** (Specify the output path of hbir.) –

- **march** (Specify march to export hbir.) – Valid options are bayes（J5处理器使用） and bernoulli2（X3处理器使用）. If not provided, use horizon plugin global march.

**返回**

**返回类型**

input names and output names

```python
horizon_plugin_pytorch.quantization.perf_model(module: Union[torch.jit._script.ScriptModule, torch.nn.modules.module.Module], example_inputs: tuple, march: Optional[str] = None, out_dir: str = '.', name: Optional[str] = None, hbm: Optional[str] = None, input_source: Union[Sequence[str], str] = 'ddr', input_layout: Optional[str] = None, output_layout: str = 'NCHW', opt: Union[str, int] = 'O3', balance_factor: int = 2, progressbar: bool = True, jobs: int = 16, layer_details: bool = False, extra_args: Optional[list] = None)
```
Estimate the performance of nn.Module or jit.ScriptModule.

**参数**

- **module** (nn.Module or jit.ScriptModule.) –

- **example_inputs** (A tuple of example inputs, in torch.tensor format.) – For jit.trace and shape inference.

- **march** (Specify the target march of bpu.) – Valid options are bayes（J5处理器使用） and bernoulli2（X3处理器使用）. If not provided, use horizon plugin global march.

- **out_dir** (Specify the output directry to hold the performance results.) –

- **name** (Name of the model, recorded in hbm.) – Can be obtained by hbdk-disas or hbrtGetModelNamesInHBM in runtime.

- **hbm** (Specify the output path of hbdk-cc.) –

- **input_source** (Specify input features' sources(ddr/resizer/pyramid)) –

- **input_layout** (Specify input layout of all model inputs.) – Available layouts are NHWC, NCHW, BPU_RAW.

- **output_layout** (Specify input layout of all model inputs.) – Available layouts are NHWC, NCHW, BPU_RAW.

- **opt** (Specify optimization options.) – Available options are O0, O1, O2, O3, ddr, fast, balance.

- **balance_factor** (Specify the balance ratio when optimization options is) – ‘balance’.

- **progressbar** (Show compilation progress to alleviate anxiety.) –

- **jobs** (Specify number of threads launched during compiler optimization.) – Default is ‘16’. 0 means use all available hard#ware concurrency.

- **layer_details** (show layer performance details. (dev use only)) –

- **extra_args** (specify extra args listed in "hbdk-cc -h".) – format in list of string: e.g. [‘–ability-entry’, str(entry_value), …]

**返回**

**返回类型**

Performance details in json dict. Or error code when fail.

```python
horizon_plugin_pytorch.quantization.visualize_model(module: Union[torch.jit._script.ScriptModule, torch.nn.modules.module.Module], example_inputs: tuple, march: Optional[str] = None, save_path: Optional[str] = None, show: bool = True)
```
Visualize nn.Module or jit.ScriptModule at the view of HBDK.

**参数**

- **module** (nn.Module or jit.ScriptModule.) –

- **example_inputs** (A tuple of example inputs, in torch.tensor format.) – For jit.trace and shape inference.

- **march** (Specify the target march of bpu.) – Valid options are bayes（J5处理器使用） and bernoulli2（X3处理器使用）. If not provided, use horizon plugin global march.

- **save_path** (Specify path to save the plot image.) –

- **show** (Display the plotted image via display.) – Make sure X-server is correctly configured.

**返回**

**返回类型**

None

## Horizon 算子 API

```python
horizon_plugin_pytorch.nn.functional.filter(*inputs: Union[Tuple[torch.Tensor], Tuple[horizon_plugin_pytorch.qtensor.QTensor]], threshold: float, idx_range: Optional[Tuple[int, int]] = None) → List[List[torch.Tensor]]
```
Filter.

The output order is different with bpu, because that the compiler do some optimization and slice input following complex rules, which is hard to be done by plugin.

All inputs are filtered along HW by the max value within a range in channel dim of the first input. Each NCHW input is splited, transposed and flattened to List[Tensor[H * W, C]] first. If input is QTensor, the output will be dequantized.

**参数**

- **inputs** (Union[Tuple[Tensor], Tuple[QTensor]]) – Data in NCHW format. Each input shold have the same size in N, H, W. The output will be selected according to the first input.

- **threshold** (float) – Threshold, the lower bound of output.

- **idx_range** (Optional[Tuple[int, int]], optional) – The index range of values counted in compare of the first input. Defaults to None which means use all the values.

**返回**

A list with same length of batch size, and each element contains: max_value: Flattened max value within idx_range in channel dim. max_idx: Flattened max value index in channel dim. coord: The original coordinates of the output data in the input data in the shape of [M, (h, w)]. (multi) data: Filtered data in the shape of [M, C].

**返回类型**

Union[List[List[Tensor]], List[List[QTensor]]]

```python
class horizon_plugin_pytorch.nn.detection_post_process.DetectionPostProcess(score_threshold=0, regression_scale=None, background_class_idx=None, size_threshold=None, image_size=None, pre_decode_top_n=None, post_decode_top_n=None, iou_threshold=None, pre_nms_top_n=None, post_nms_top_n=None, nms_on_each_level=False, mode='normal')
```
General post process for object detection models.

Compatible with YOLO, SSD, RetinaNet, Faster-RCNN (RPN & RCNN), etc. Note that this is a float OP, please use after DequantStubs.

**参数**

- **score_threshold** (int, optional) – Filter boxes whose score is lower than this. Defaults to 0.

- **regression_scale** (Tuple[float, float, float, float], optional) – Scale to be multiplyed to box regressions. Defaults to None.

- **background_class_idx** (int, optional) – Specify the class index to be ignored. Defaults to None.

- **size_threshold** (float, optional) – Filter bixes whose height or width smaller than this. Defaults to None.

- **image_size** (Tuple[int, int], optional) – Clip boxes to image sizes. Defaults to None.

- **pre_decode_top_n** (int, optional) – Get top n boxes by objectness (first element in the score vector) before decode. Defaults to None.

- **post_decode_top_n** (int, optional) – Get top n boxes by score after decode. Defaults to None.

- **iou_threshold** (float, optional) – IoU threshold for nms. Defaults to None.

- **pre_nms_top_n** (int, optional) – Get top n boxes by score before nms. Defaults to None.

- **post_nms_top_n** (int, optional) – Get top n boxes by score after nms. Defaults to None.

- **nms_on_each_level** (bool, optional) – Whether do nms on each level seperately. Defaults to False.

- **mode** (str, optional) – Only support ‘normal’ and ‘yolo’. If set to ‘yolo’: 1. Box will be filtered by objectness rathen than classification scores. 2. dx, dy in regressions will be treated as absolute offset. 3. Objectness will be multiplyed to classification scores. Defaults to ‘normal’.


```python
forward(boxes: List[torch.Tensor], scores: List[torch.Tensor], regressions: List[torch.Tensor], image_shapes: Optional[torch.Tensor] = None) → Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor]]
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

:::info 小技巧
Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.
:::

```python
class horizon_plugin_pytorch.nn.bgr_to_yuv444.BgrToYuv444(channel_reversal=False)
```
Convert image color format from bgr to yuv444.

**参数**

- **channel_reversal** (bool, optional) – Color channel order, set to True when used on RGB input. Defaults to False.

**forward(input)**

Defines the computation performed at every call.

Should be overridden by all subclasses.

:::info 小技巧
Although the recipe for forward pass needs to be defined within this function, one should call the Module instance afterwards instead of this since the former takes care of running the registered hooks while the latter silently ignores them.
:::

```python
class horizon_plugin_pytorch.nn.detection_post_process_v1.DetectionPostProcessV1(num_classes: int, box_filter_threshold: float, class_offsets: List[int], use_clippings: bool, image_size: Tuple[int, int], nms_threshold: float, pre_nms_top_k: int, post_nms_top_k: int, nms_padding_mode: Optional[str] = None, nms_margin: float = 0.0, use_stable_sort: Optional[bool] = None, bbox_min_hw: Tuple[float, float] = (0, 0))
```
Post process for object detection models. Only supported on bernoulli2.

This operation is implemented on BPU, thus is expected to be faster than cpu implementation. This operation requires input_scale = 1 / 2 ** 4, or a rescale will be applied to the input data. So you can manually set the output scale of previous op (Conv2d for example) to 1 / 2 ** 4 to avoid the rescale and get best performance and accuracy.

Major differences with DetectionPostProcess:

1. Each anchor will generate only one pred bbox totally, but in DetectionPostProcess each anchor will generate one bbox for each class (num_classes bboxes totally).

2. NMS has a margin param, box2 will only be supressed by box1 when box1.score - box2.score > margin (box1.score > box2.score in DetectionPostProcess).

3. A offset can be added to the output class indices ( using class_offsets).

**参数**

- **num_classes** (int) – Class number.

- **box_filter_threshold** (float) – Default threshold to filter box by max score.

- **class_offsets** (List[int]) – Offset to be added to output class index for each branch.

- **use_clippings** (List[bool]) – Whether clip box to image size. If input is padded, you can clip box to real content by providing image size.

- **image_size** (Tuple[int, int]) – Fixed image size in (h, w), set to None if input have different sizes.

- **nms_threshold** (float) – IoU threshold for nms.

- **nms_margin** (float) – Only supress box2 when box1.score - box2.score > nms_margin

- **pre_nms_top_k** – Maximum number of bounding boxes in each image before nms.

- **post_nms_top_k** – Maximum number of output bounding boxes in each image.

- **nms_padding_mode** – The way to pad bbox to match the number of output bounding bouxes to post_nms_top_k, can be None, “pad_zero” or “rollover”.

- **bbox_min_hw** – Minimum height and width of selected bounding boxes.

```python
forward(data: List[torch.Tensor], anchors: List[torch.Tensor], image_sizes=None) → torch.Tensor
```
Forward pass of ~DetectionPostProcessV1.

**参数**

- **data** (List[Tensor]) – (N, (4 + num_classes) * anchor_num, H, W)

- **anchors** (List[Tensor]) – (N, anchor_num * 4, H, W)

- **image_sizes** (Tensor[batch_size, (h, w)], optional) – Defaults to None.

**返回**

list of (bbox (x1, y1, x2, y2), score, class_idx).

**返回类型**

List[Tuple[Tensor, Tensor, Tensor]]


```python
horizon_plugin_pytorch.functional.centered_yuv2bgr(input: horizon_plugin_pytorch.qtensor.QTensor, swing: str = 'studio', mean: Union[List[float], torch.Tensor] = (128.0,), std: Union[List[float], torch.Tensor] = (128.0,), q_scale: Union[float, torch.Tensor] = 0.0078125) → horizon_plugin_pytorch.qtensor.QTensor
```
Convert color space.

Convert images from centered YUV444 BT.601 format to transformed and quantized BGR. Only use this operator in the quantized model. Insert it after QuantStub. Pass the scale of QuantStub to the q_scale argument and set scale of QuantStub to 1 afterwards.

**参数**

- **input** (QTensor) – Input images in centered YUV444 BT.601 format, centered by the pyramid with -128.

- **swing** (str, optional) – “studio” for YUV studio swing (Y: -112~107, U, V: -112~112). “full” for YUV full swing (Y, U, V: -128~127). default is “studio”

- **mean** (List[float] or Tensor, optional) – BGR mean, a list of float, or torch.Tensor, can be a scalar [float], or [float, float, float] for per-channel mean.

- **std** (List[float] or Tensor, optional) – BGR standard deviation, a list of float, or torch.Tensor, can be a scalar [float], or [float, float, float] for per-channel std.

- **q_scale** (float or Tensor, optional) – BGR quantization scale.

**返回**

Transformed and quantized image in BGR color, dtype is qint8. # noqa: E501

**返回类型**

QTensor

```python
horizon_plugin_pytorch.functional.centered_yuv2rgb(input: horizon_plugin_pytorch.qtensor.QTensor, swing: str = 'studio', mean: Union[List[float], torch.Tensor] = (128.0,), std: Union[List[float], torch.Tensor] = (128.0,), q_scale: Union[float, torch.Tensor] = 0.0078125) → horizon_plugin_pytorch.qtensor.QTensor
```
Convert color space.

Convert images from centered YUV444 BT.601 format to transformed and quantized RGB. Only use this operator in the quantized model. Insert it after QuantStub. Pass the scale of QuantStub to the q_scale argument and set scale of QuantStub to 1 afterwards.

**参数**

- **input** (QTensor) – Input images in centered YUV444 BT.601 format, centered by the pyramid with -128.

- **swing** (str, optional) – “studio” for YUV studio swing (Y: -112~107, U, V: -112~112). “full” for YUV full swing (Y, U, V: -128~127). default is “studio”

- **mean** (List[float] or Tensor, optional) – RGB mean, a list of float, or torch.Tensor, can be a scalar [float], or [float, float, float] for per-channel mean.

- **std** (List[float] or Tensor, optional) – RGB standard deviation, a list of float, or torch.Tensor, can be a scalar [float], or [float, float, float] for per-channel std.

- **q_scale** (float or Tensor, optional) – RGB quantization scale.

**返回**

Transformed and quantized image in RGB color, dtype is qint8. # noqa: E501

**返回类型**

QTensor

## 支持的算子

### 支持的 torch 算子

下表 op 如无特殊说明均有以下限制：输入输出数据类型 int8，input_shape: [N, C, H, W]，input_size <1G bytes，1<=N<=4096，1<=H, W, C<=65536，Feature 维度为4。如表格中添加额外限制信息，以表格中为准。

<!-- Here is a table generator for markdown:
https://www.tablesgenerator.com/markdown_tables
Steps to edit this table:
1. Copy the followed markdown code of this table
2. Go to the url, Click File -> Paste table data, paste the copyed code
3. Edit the table -> Genetate -> Copy to clipboard
4. Paste the genetated code here
 -->

<!-- table begin -->

| Torch ops | 准备浮点模型时需替换为 | bernoulli2 支持 | bayes 支持 |
|---|---|---|---|
| torch.add | torch.nn.quantized.FloatFunctional 或<br/>horizon.nn.quantized.FloatFunctional | 支持。QAT 有训练参数，不能单独在预测中使用。in_channel<=2048 | 支持。QAT 有训练参数，不能单独在预测中使用。输入输出支持 int8/int16. 支持除 N 维以外的广播，只能有一个 input 广播。 |
| torch.sub | horizon.nn.quantized.FloatFunctional | 支持。QAT 有训练参数，不能单独在预测中使用。in_channel<=2048 | 支持。QAT 有训练参数，不能单独在预测中使用。输入输出支持 int8/int16. 支持除 N 维以外的广播，只能有一个 input 广播。 |
| torch.mul | torch.nn.quantized.FloatFunctional 或<br/>horizon.nn.quantized.FloatFunctional | 支持。QAT 有训练参数，不能单独在预测中使用。in_channel<=2048 | 支持。QAT 有训练参数，不能单独在预测中使用。输入输出支持 int8/int16. 支持除 N 维以外的广播，只能有一个 input 广播。 |
| torch.sum | horizon.nn.quantized.FloatFunctional | 只支持 batch 和 channel 方向的 sum。QAT 有 训练参数，不要单独在预测中使用。 | 支持。QAT 有训练参数，不能单独在预测中使用。输入输出支持 int8/int16. 仅支持 HWC 三个维度的 sum |
| torch.matmul | horizon.nn.quantized.FloatFunctional | 支持。对于 matmul(a, b)，可通过设置参数对  b 在内部进行转置 | 支持。输入 int8, 输出 int8/int16/int32. 对于 matmul(a, b)， 可通过设置参数对 b 在内部进行转置。<br/>input shape: [N, C, H, W],  input_size<1 G bytes, N<=4096, C, H, W<=8192. |
| torch.cat | torch.nn.quantized.FloatFunctional 或<br/>horizon.nn.quantized.FloatFunctional | 对于 cat(a, b), a 和 b 相差不能太大，否则 会出现其中一个操作数吃掉另外一个操作数的现象。<br/>QAT 有训练参数，不要单独在预测中使用。 | 对于 cat(a, b), a 和 b 相差不能太大，否则 会出现其中一个操作数吃掉另外一个操作数的现象。  <br/>QAT 有训练参数，不要单独在预测中使用。<br/>input shape: [N, C, H, W],  N<=4096, HWC<=65536, 2<=input number<=1024 |
| torch.add_scalar | torch.nn.quantized.FloatFunctional 或<br/>horizon.nn.quantized.FloatFunctional | 不支持 | 支持一个输入int8/int16的Tensor，另一个输入为标量 |
| torch.mul_scalar | torch.nn.quantized.FloatFunctional 或<br/>horizon.nn.quantized.FloatFunctional | 不支持 | 支持一个输入int8/int16的Tensor，另一个输入为标量 |
| torch.maximum | horizon.nn.quantized.FloatFunctional | 不支持 | 支持输入输出int8/int16 |
| torch.minimum | horizon.nn.quantized.FloatFunctional | 不支持 | 支持输入输出int8/int16 |
| torch.mean | horizon.nn.quantized.FloatFunctional | 只支持在 channel 方向的 mean。QAT 有训练参数，不要单独在预测中使用。 | 支持在 CHW 上求 mean.QAT 有量化参数，不能单独在预测中使用。支持输入输出 int8/int16 |
| torch.sqrt | horizon.nn.Sqrt | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.atan | horizon.nn.Atan | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.sin | horizon.nn.Sin | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.cos | horizon.nn.Cos | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.clamp/clip, tensor.clamp/clip |  | 不支持 | 支持min和max的输入为Tensor/常量Tensor/标量/None。<br/>为常量Tensor时，min 和 max 的输入数据范围最好和 input 一致，否则有精度风险 |
| torch.pow | horizon.nn.Pow | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.max |  | 支持。只能作为模型输出。<br/>输出格式和 torch 不同:<br/>编译器支持的输 出是一个 Tensor， 其中一个 channel 中的值是 max_value，<br/>另一个 channel 中的值是max_value_index | 支持。<br/>输入支持int8/int16, 同时输出 int8/16 value 和 int32 index，index 只能作为模型输出。<br/>input_shape: [N, C, H, W], 1<=N<=4096, 1<=H, W, C<=65535 |
| torch.min |  | 不支持 | 支持。限制同 torch.max |
| tensor.max |  | 参考 torch.max | 参考 torch.max |
| tensor.min |  | 不支持 | 参考 torch.min |
| torch.split |  | 支持 | 支持输入输出int8/int16 |
| torch.eq |  | 不支持 | 支持输入输出int8/int16 |
| torch.ge |  | 不支持 | 支持输入输出int8/int16 |
| torch.greater |  | 不支持 | 支持输入输出int8/int16 |
| torch.greater_equal |  | 不支持 | 支持输入输出int8/int16 |
| torch.gt |  | 不支持 | 支持输入输出int8/int16 |
| torch.le |  | 不支持 | 支持输入输出int8/int16 |
| torch.less |  | 不支持 | 支持输入输出int8/int16 |
| torch.less_equal |  | 不支持 | 支持输入输出int8/int16 |
| torch.lt |  | 不支持 | 支持输入输出int8/int16 |
| tensor.eq |  | 不支持 | 支持输入输出int8/int16 |
| tensor.ge |  | 不支持 | 支持输入输出int8/int16 |
| tensor.greater |  | 不支持 | 支持输入输出int8/int16 |
| tensor.greater_equal |  | 不支持 | 支持输入输出int8/int16 |
| tensor.gt |  | 不支持 | 支持输入输出int8/int16 |
| tensor.le |  | 不支持 | 支持输入输出int8/int16 |
| tensor.less |  | 不支持 | 支持输入输出int8/int16 |
| tensor.less_equal |  | 不支持 | 支持输入输出int8/int16 |
| tensor.expand |  | 不支持 | 支持输入输出int8/int16 |
| tensor.repeat |  | 不支持 | 支持输入输出int8/int16 |
| tensor.tile |  | 不支持 | 支持输入输出int8/int16 |
| torch.nn.GLU |  | 不支持 | 支持输入输出int8/int16 |
| torch.nn.GELU |  | 不支持 | 支持输入输出int8/int16 |
| torch.nn.PReLU |  | 不支持 | 支持输入输出int8/int16 |
| torch.nn.LeakyReLU |  | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.nn.LSTMCell |  | 不支持 | 支持输入输出int8/int16。输入是 2 维 |
| torch.argmax |  | 参考 torch.max | 参考 torch.max |
| torch.argmin |  | 参考 torch.min | 参考 torch.min |
| tensor.argmax |  | 参考 torch.max | 参考 torch.max |
| tensor.argmin |  | 参考 torch.min | 参考 torch.min |
| tensor.reshape |  | 只支持 H 和 W 方向的 reshape | dim <= 10, 1 <= each_dim_size < 65536 |
| torch.nn.Tanh |  | 支持，底层查表实现，有一定精度风险 | 支持输入输出int8/int16 |
| torch.nn.ReLU |  | Conv2d+BN+ReLU 这种模式会自动 fuse 成 BpuConv2d，<br/>否则，单独跑 ReLU.ReLU 对量化训练不友好，<br/>建议用户使用 ReLU6.在 QAT 阶段，会默认使用 relu6 | Conv2d+BN+ReLU 这种模式会自动 fuse 成 BpuConv2d，否则，单独跑 ReLU.ReLU 对量化训练不友好， <br/>建议用户使用 ReLU6.<br/>在 QAT 阶段，会默认使用 relu6 |
| torch.nn.SiLU |  | 支持，底层查表实现，有一定精度风险 | 支持 |
| torch.nn.Conv2d |  | 支持。kernel<=7.channel(one group) <= 2048. dilation=(1, 1)/(2, 2)/(4, 4)，<br/>当 dilation!=(1, 1)时,  stride必须为(1, 1). HxWxC <= 32768 | 支持。out_channel<=8192，作为模型输出时，out_channel <= 16384. <br/>输入 channel<=8192, kernel<32, dilation<=16, 当dilation!=1时, stride只能 为1.<br/>支持 sumin, 带 sumin 的 conv 只支持 stride 为 (1, 1) 或 (2, 2). <br/>weight_shape: [N, C, H, W], N, C<=8192, H, W<=31, 作为模型输出C<=16384, weight_size < 65535.<br/>padding<=256.支持int8/int16输入，int8/int16/int32输出。 <br/>int16输入时累加和不能超过 int32 范围 |
| torch.nn.Linear |  | 不支持 | 支持。in_features <= 8192, out_features <= 8192. |
| torch.nn.Conv3d |  | 不支持 | input: [N, C, D, H, W] int8, N<=128; <br/>H, W, D, C<=65536; <br/>weight: [C_o, C_i, D, H, W] int8, N, C<=65536, D, H<=9, W<=8191; <br/>bias: int32; output: [N, C, D, H, W] int8, int16, int32; <br/>stride: [D, H, W], D, H, W 等于 1 或 2, 并且 D, H, W 相同; <br/>padding: [D, H, W], D<=kernel_d/2, H<=kernel_h/2, W<=kernel_w/2(kernel_w 指 weight W 维大小) group, <br/>dilation: 暂不支持 |
| tensor.transpose |  | 不支持 | 支持输入输出 int8, int16, int32.不支持对 N 维的 transpose |
| torch.nn.Sigmoid |  | 支持，底层查表实现，有一定精度风险 | 支持输入输出int8/int16 |
| torch.nn.Dropout |  | 训练 op，不体现在预测模型中 | 训练 op，不体现在预测模型中 |
| torch.nn.Softmax |  | 不支持 | 支持输入输出int8/int16，底层查表实现，有一定精度风险 |
| torch.nn.Identity |  | 训练 op，不体现在预测模型中 | 训练 op，不体现在预测模型中 |
| torch.nn.AvgPool2d |  | 1<=kernel<=7，1<=stride<=185 | 支持。1<=kernel, stride, padding<=256; |
| torch.nn.MaxPool2d |  | 1<=kernel<=64, 1<=stride<=256, padding>=0 | input_shape: [N, C, H, W], 1<=H, W, C<=8192;1<=kernel, stride<=256; 0<=padding<=255; |
| torch.nn.ZeroPad2d |  | 支持。 | 支持输入输出 int8/int16. |
| torch.nn.Dropout2d |  | 训练 op，不体现在预测模型中 | 训练 op，不体现在预测模型中 |
| torch.nn.Layernorm |  | 不支持 | 输出支持int8/int16, 底层查表实现，有精度风险. <br/>可通过 `rsqrt_kwargs` 属性来控制内部 rsqrt 查表的参数， <br/>若遇到 convert 精度降低的问题可以尝试 `layernorm_op.rsqrt_kwargs = {"auto_divide_strategy": "curvature"}`. <br/>H * W <= 16384, normalized_shape H * W < 16384 |
| torch.nn.BatchNorm2d |  | BatchNorm2d 在 QAT 阶段被吸收，不体现在预测模型中。<br/>由于编译器限制，独立使用的 BatchNorm2d 底层调用 BpuConvolution 实现 | BatchNorm2d 在 QAT 阶段被吸收，因此，不体现在模型中。独立使用限制参考 Conv2d |
| Tensor.\_\_getitem\_\_ |  | 支持 | 支持 |
| torch.nn.ConstantPad2d |  | 支持 | 支持，限制参考 ZeroPad2d |
| torch.nn.SyncBatchNorm |  | 训练 op，不体现在预测模型中 | 训练 op，不体现在预测模型中 |
| torch.nn.ChannelShuffle |  | 支持 | 支持输入输出int8/int16. shuffle_index 中的数值不能重复 |
| torch.nn.PixelShuffle |  | 支持输入输出int8/int16 | 支持输入输出int8/int16 |
| torch.nn.PixelUnshuffle |  | 支持输入输出int8/int16 | 支持输入输出int8/int16 |
| torch.nn.ConvTranspose2d |  | 支持。2<=kernel<= 14.channel<=2048. <br/>padding H*W=[0, (kernel_h-1)/2] * [0, (kernel_w-1)/2] 2<=stride<=4, dilation=(1, 1) | 支持。输入shape: [N, C, H, W], 1<=N<=128, 1<=channel<=2048; <br/>weight_shape: [N, C, H, W], 1<=N, C<=2048, 2<=H, W<=14, weight_size<=65535;<br/>kernel>=stride, 1<=stride<=14, 1<=out_channel<=2048, in_channel<=2048 <br/>pad<=kernel/stride, 0<=out_pad<=1;<br/>bias 类型为 int32;<br/>支持sumin, sumin输入类型为int8;<br/>0<=output_padding<=1;<br/>支持group, 要求weight_n和 输入channel均能被group整除;<br/>dilation=1 |
| torch.nn.Upsample |  | 支持。参考 torch.nn.functional.interpolate | 参考 torch.nn.functional.interpolate |
| torch.nn.UpsamplingNearest2d |  | 支持。限制参考 torch.nn.functional.interpolate | 支持。限制参考 torch.nn.functional.interpolate |
| torch.nn.UpsamplingBilinear2d |  | 支持。限制参考 torch.nn.functional.interpolate | 支持。限制参考 torch.nn.functional.interpolate |
| torch.nn.functional.pad |  | 支持除 `reflect` 外的其他模式 | 支持除 `reflect` 外的其他模式 |
| torch.nn.functional.relu | torch.nn.ReLU | Conv2d+BN+ReLU 这种模式会自动 fuse 成 BpuConv2d，<br/>否则，单独跑 ReLU.ReLU 对量化 训练不友好，建议用户使用 ReLU6.在 QAT 阶段，会默认使用 relu6 | Conv2d+BN+ReLU 这种模式会自动 fuse 成 BpuConv2d，否则，单独跑 ReLU.ReLU 对量化训练不友好， <br/>建议用户使用 ReLU6.<br/>在 QAT 阶段，会默认使用 relu6 |
| torch.nn.functional.relu6(fused) | torch.nn.ReLU6 | Conv2d+BN+ReLU6 这种模式会自动 fuse 成 BpuConv2d，否则，单独跑 ReLU6 | Conv2d+BN+ReLU6 这种模式会自动 fuse 成 BpuConv2d，否则，单独跑 ReLU6 |
| torch.nn.ReplicationPad2d |  | 支持。 | 支持。限制参考 ZeroPad2d |
| torch.quantization.QuantStub | horizon.quantization.QuantStub | 支持。<br/>典型使用场景：整个网络模型的输入。 <br/>模型分段的场景：数据从 CPU 送入到 BPU 之前需要把数据进行量化。<br/>scale 参数设置方法：scale 的设置和具体的输入有关。设置目标是使得输入的 float 类型的数据尽量<br/>高精度地量化到 int8 类型<br/>这就有两个方面的要求：可以覆盖所有的（至少是绝大部分）输入数据，量化精度高。<br/>例如：输入 float 的范围是 (-1, 1), 那么，我们可以设置 scale = 1 / 128。<br/>Float 预训练模型：在预训练模型中，由于模型已经训练好，不一定遵循上述 scale 参数设置方法，<br/>这时，可以通过插入一个特殊的 conv 的方法来解决。要求输入 QuantStub 的数据的分布是均匀的 | 支持。典型使用场景：整个网络模型的输入。 <br/>模型分段的场景，数据从 CPU 送入到 BPU 之前需要把数据进行量化。<br/>scale 参数设置方法：scale 的设置和具体的输入有关。 <br/>设置目标是使得输入的 float 类型的数据尽量 高精度地量化到 int8 类型，<br/>这就有两个方面的要求：可以覆盖所有的（至少是绝大部分）输入数据，  <br/>量化精度高。<br/>例如：输入 float 的范围是 (-1, 1),   那么，我们可以设置 scale = 1 / 128。<br/>Float 预训练模型：在预训练模型中，由于模型已经训练好，不一定遵循上述 scale 参数设置方法， <br/>这时，可以通过插入一个特殊的 conv 的方法来解决。要求输入 QuantStub 的数据的分布是均匀的 |
| torch.quantization.DeQuantStub |  | 典型使用场景：网络模型分段的场景，需要把数据 从 BPU 传输到 CPU，在 CPU 上进行反量化， <br/>方便 CPU 上处理 | 典型使用场景：网络模型分段的场景，需要把数据 从 BPU 传输到 CPU，在 CPU 上进行反量化，方便 CPU 上处理 |
| torch.nn.functional.interpolate | horizon.nn.Interpolate | 只支持 nearest 和 billinear 插值模式。1/256<缩放比例<=256 | 只支持 nearest 和 billinear 插值模式。input_shape: [N, C, H, W], 1<=C, H, W<=8192 |
| torch.nn.functional.grid_sample |  | 不支持 | 支持。输入shape: [N, C, H, W], 1<=H, W<=1024 且 H*W<=512*1024; <br/>grid 只支持 qint16，只支持 bilinear 和 nearest 插值<br/>padding 模式只支持 zeros 和 border; |
| torch 中没有 Correlation 算子 | horizon.nn.Correlation | 不支持 | 支持。输入shape: [N, C, H, W], 1<=C<=1024, <br/>kernel 必须为奇数, <br/>min(H, W) + 2 * pad - 2 * (max_displacement/stride2*stride2)-kernel+1 >= stride1 |
| torch.log | horizon.nn.HardLog | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险 |
| torch.masked_fill |  | 不支持 | 支持输入输出 int8 |
| torch 中没有 PointPillarsScatter | horizon.nn.PointPillarsScatter | 不支持 | 支持输入输出int8/int16，需要三个输入，<br/>三个输入的 shape 分别是: [M, C],  [M, 4]和[4]，<br/>第3个输入表示该算子输出feature的 shape: (N, C, H, W) |
| torch.div | horizon.nn.Div | 不支持 | 支持输入输出 int16 |
| torch 中没有 bgr2centered_yuv 算子 | horizon.bgr2centered_yuv | 不支持 | 支持。用于数据预处理。将 0~255 的 BGR 图片转换为 centered YUV 格式 |
| torch 中没有 rgb2centered_yuv 算子 | horizon.rgb2centered_yuv | 不支持 | 支持。用于数据预处理。将 0~255 的 RGB 图片转换为 centered YUV 格式 |
| torch 中没有 centered_yuv2bgr 算子 | horizon.centered_yuv2bgr | 不支持 | 支持输入输出 int8。当用户使用 BGR 图片训练 QAT 模型时，<br/>在定点模型中插入该算子将 YUV 格式的输入图片转换成 GBR 格式 |
| torch 中没有 centered_yuv2rgb 算子 | horizon.centered_yuv2rgb | 不支持 | 支持输入输出 int8。当用户使用 RGB 图片训练 QAT 模型时，<br/>在定点模型中插入该算子将 YUV 格式的输入图片转换成 RGB 格式 |
| torch.nn.MultiheadAttention |  | 不支持 | 不支持 add_bias_kv、add_zero_attn 和 q k v embed_dim 不一致的情况，<br/>支持输入输出 int8/int16，底层查表算子与 mask 量化可能带来精度风险 |
| torch.reciprocal | horizon.nn.Reciprocal | 不支持 | 支持输入输出int8/int16 |
| torch 中没有 rcnn_post_process 算子 | horizon.nn.RcnnPostProcess | 不支持 | 该算子推理时需在CPU上运算，支持浮点输入输出，用于对RCNN的输出进行包括NMS在内的一系列后处理 |
| torch.topk |  | 不支持 | 支持输入输出int8/int16/int32。|
| torch.gather |  | 不支持 | 支持输入输出int8/int16/int32。|
| torch.abs | horizon.abs | 不支持 | 支持输入输出int8/int16。|
| torch.nn.Softplus |  | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险。|
| torch.nn.ELU |  | 不支持 | 支持输入输出int8/int16。底层查表实现，有精度风险。|
| torch.ceil | horizon.nn.Ceil | 不支持 | 支持输入输出int8/int16。int8下输入数量级不要超过1e6, int16下输入数量级不要超过1e8。 |
| torch.floor | horizon.nn.Floor | 不支持 | 支持输入输出int8/int16。int8下输入数量级不要超过1e6, int16下输入数量级不要超过1e8。 |

### 支持的 torchvision 算子

|  torchvision 算子                                 | 准备浮点模型时需替换为        | bernoulli2 支持                               |  bayes 支持     |
|---------------------------------------------------|-------------------------------|-----------------------------------------------|-----------------|
|  torchvision.ops.RoIAlign                         |                               | 支持                                          | 支持。1<=feature number<=5;bbox 仅支持 `List[Tensor]` 格式 shape:[1, box_num, 4],<br/>bbox 最后一维 4 个数分别为: [left, top, right, bottom] |
|  torchvision.ops.MultiScaleRoIAlign               | horizon.nn.MultiScaleRoIAlign | 支持                                          | 支持。限制信息参考 RoIAlign                            |
|  torchvision.models.detection.rpn.AnchorGenerator | horizon.nn.AnchorGenerator    | 仅支持 Tensor.shape 可以离线确定的情况        | 支持输入 int8/int16/int32/float32, 输出 float32       |
<!--
|  torchvision.ops.DeformConv2d                     |                               | 不支持                                        | 编译器暂未支持。                                      |
-->


