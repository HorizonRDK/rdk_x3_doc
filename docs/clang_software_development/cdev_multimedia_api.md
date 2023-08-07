---
sidebar_position: 2
---

# 5.2 多媒体接口说明

## VIO（视频输入）API

`VIO` 模块提供操作 `MIPI` 摄像头和操作图像处理的功能。

`VIO` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_init_vio_module | **初始化VIO对象** |
| sp_release_vio_module | **销毁VIO对象** |
| sp_open_camera | **打开摄像头** |
| sp_open_camera_v2 | **指定分辨率打开摄像头** |
| sp_open_vps | **打开VPS** |
| sp_vio_close | **关闭摄像头** |
| sp_vio_get_frame | **获取视频图像帧** |
| sp_vio_set_frame | **发送视频图像帧给vps模块** |


### sp_init_vio_module  

**【函数原型】**  

`void *sp_init_vio_module()`

**【功能描述】**  

初始化`VIO`对象，创建操作句柄。在其他接口调用前必须执行。

**【参数】**

无

**【返回类型】**  

成功返回一个`VIO`对象指针，失败返回`NULL`

### sp_release_vio_module  

**【函数原型】**  

`void sp_release_vio_module(void *obj)`

**【功能描述】**  

销毁`VIO`对象。

**【参数】**

- `obj`： 调用初始化接口时得到的`VIO`对象指针。

**【返回类型】**  

无

### sp_open_camera  

**【函数原型】**  

`int32_t sp_open_camera(void *obj, const int32_t pipe_id, const int32_t video_index, int32_t chn_num, int32_t *width, int32_t *height)`

**【功能描述】**  

初始化接入到RDK X3上的MIPI摄像头。
设置输出分辨率，支持设置最多5组分辨率，其中只有1组可以放大，4组可以缩小。最大支持放大到原始图像的1.5倍，最小支持缩小到原始图像的1/8。

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `pipe_id`：支持多组数据输入，建议填0
- `video_index`：camera对应的host编号，-1表示自动探测，编号可以查看 /etc/board_config.json 配置文件
- `chn_num`：设置输出多少种不同分辨率的图像，最大为5，最小为1。
- `width`：配置输出宽度的数组地址
- `height`：配置输出高度的数组地址

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_open_camera_v2  

**【函数原型】**  

`int32_t sp_open_camera_v2(void *obj, const int32_t pipe_id, const int32_t video_index, int32_t chn_num, sp_sensors_parameters *parameters, int32_t *width, int32_t *height)`

**【功能描述】**  

初始化接入到RDK X3上的MIPI摄像头。  
支持指定摄像头原始输出RAW的分辨率大小，通过`sp_sensors_parameters`设置。  
支持设置输出分辨率，支持设置最多5组分辨率，其中只有1组可以放大，4组可以缩小。最大支持放大到原始图像的1.5倍，最小支持缩小到原始图像的1/8。

目前支持的摄像头分辨率见下表：

| camera | 分辨率 |
| ---- | ----- |
|IMX219|1920x1080@30fps(default), 640x480@30fps, 1632x1232@30fps, 3264x2464@15fps(max)|
|IMX477|1920x1080@50fps(default), 1280x960@120fps, 2016x1520@40fps, 4000x3000@10fps(max)|
|OV5647|1920x1080@30fps(default), 640x480@60fps, 1280x960@30fps, 2592x1944@15fps(max)|
|F37|1920x1080@30fps(default)|
|GC4663|2560x1440@30fps(default)|

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `pipe_id`：支持多组数据输入，建议填0
- `video_index`：camera对应的host编号，-1表示自动探测，编号可以查看 /etc/board_config.json 配置文件
- `chn_num`：设置输出多少种不同分辨率的图像，最大为5，最小为1。
- `parameters`：camera RAW输出相关结构体，用于指定分辨率和帧率
- `width`：配置输出宽度的数组地址
- `height`：配置输出高度的数组地址

`sp_sensors_parameters`结构体成员见下表：

| 数据类型 | 成员 | 注释 |
| ---- | ----- | ----- |
|int32_t|raw_height|摄像头输出RAW的高度|
|int32_t|raw_width|摄像头输出RAW的宽度|
|int32_t|fps|摄像头输出的帧率|


**【返回类型】** 

成功返回 0，失败返回 -1

### sp_open_vps  

**【函数原型】**  

`int32_t sp_open_vps(void *obj, const int32_t pipe_id, int32_t chn_num, int32_t proc_mode, int32_t src_width, int32_t src_height, int32_t *dst_width, int32_t *dst_height, int32_t *crop_x, int32_t *crop_y, int32_t *crop_width, int32_t *crop_height, int32_t *rotate)`

**【功能描述】**  

打开一路图像处理模块，支持对输入的图像完成缩小、放大、旋转、裁剪任务。

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `pipe_id`：支持多次打开，通过`pipe_id`进行区分。
- `chn_num`：设置输出图像数量，最大为5，与设置的目标高宽数组大小有关
- `proc_mod`：处理模式，当前支持：`SP_VPS_SCALE` 仅缩放、`SP_VPS_SCALE_CROP` 缩放并裁剪、`SP_VPS_SCALE_ROTATE` 缩放并旋转、`SP_VPS_SCALE_ROTATE_CROP` 缩放之后旋转并裁剪
- `src_width`：原始帧宽度
- `src_height`：原始帧高度
- `dst_width`：配置目标输出宽度的数组地址
- `dst_height`：配置目标输出高度的数组地址
- `crop_x`：裁剪区域的左上角x坐标集合，当`proc_mod`没有设置裁剪功能时，传入`NULL`
- `crop_y`：裁剪区域的左上角y坐标集合，当`proc_mod`没有设置裁剪功能时，传入`NULL`
- `crop_width`：裁剪区域的宽度，当`proc_mod`没有设置裁剪功能时，传入`NULL`
- `crop_height`：裁剪区域的高度，当`proc_mod`没有设置裁剪功能时，传入`NULL`
- `rotate`：旋转角度集合，目前支持`ROTATION_90` 90°、`ROTATION_180` 180°和`ROTATION_270` 270°，当`proc_mod`没有设置旋转功能时，传入`NULL`

**【返回类型】**  

成功返回 0，失败返回 -1

### sp_vio_close  

**【函数原型】**  

`int32_t sp_vio_close(void *obj)`

**【功能描述】**  

根据传入的 `obj` 是打开的 `camera` 还是 `vps`决定关闭camera还是vps模块。

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针  

**【返回类型】**  

成功返回 0，失败返回 -1

### sp_vio_get_frame  

**【函数原型】**  

`int32_t sp_vio_get_frame(void *obj, char *frame_buffer, int32_t width, int32_t height, const int32_t timeout)`

**【功能描述】**  

获取指定分辨率的图像帧数据（分辨率在打开模块时需要传入，否则会获取失败）。返回数据格式为 `NV12` 的 `YUV` 图像。

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `frame_buffer`：已经预分配内存的buffer指针，用于保存获取出来的图片，目前获取到的图像都是`NV12`格式，所以预分配内存大小可以由公式`高 * 宽 * 3 / 2 `，也可以利用提供的宏定义 `FRAME_BUFFER_SIZE(w, h)`进行内存大小计算
- `width`：`image_buffer`保存图片的宽，必须是在`sp_open_camera`或者`sp_open_vps`配置好的输出宽
- `height`：`image_buffer`保存图片的高，必须是在`sp_open_camera`或者`sp_open_vps`配置好的输出高
- `timeout`：获取图片的超时时间，单位为`ms`，一般设置为`2000`

**【返回类型】**  

成功返回 0，失败返回 -1 

### sp_vio_get_raw  

**【函数原型】**  

`int32_t sp_vio_get_raw(void *obj, char *frame_buffer, int32_t width, int32_t height, const int32_t timeout)`

**【功能描述】**  

获取摄像头的raw图数据

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `frame_buffer`：已经预分配内存的buffer指针，用于保存获取出来的raw图，预分配内存字节大小可以由公式`(高 * 宽 * 图像深度)/8`计算得出
- `width`：获取raw图时传`NULL`
- `height`：获取raw图时传`NULL`
- `timeout`：获取图片的超时时间，单位为`ms`，一般设置为`2000`

**【返回类型】**  

成功返回 0，失败返回 -1 

### sp_vio_get_yuv  

**【函数原型】**  

`int32_t sp_vio_get_yuv(void *obj, char *frame_buffer, int32_t width, int32_t height, const int32_t timeout)`

**【功能描述】**  

获取摄像头的ISP模块的YUV数据

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `frame_buffer`：已经预分配内存的buffer指针，用于保存获取出来的图片，目前获取到的图像都是`NV12`格式，所以预分配内存大小可以由公式`高 * 宽 * 3 / 2 `，也可以利用提供的宏定义 `FRAME_BUFFER_SIZE(w, h)`进行内存大小计算
- `width`：获取ISP的YUV数据时传`NULL`
- `height`：获取ISP的YUV数据传`NULL`
- `timeout`：获取图片的超时时间，单位为`ms`，一般设置为`2000`

**【返回类型】**  

成功返回 0，失败返回 -1 

### sp_vio_set_frame  

**【函数原型】**  

`int32_t sp_vio_set_frame(void *obj, void *frame_buffer, int32_t size)`

**【功能描述】**  

在使用`vps`模块功能时，源数据需要通过调用本接口送入，`frame_buffer`里面的数据必须是 `NV12` 格式的图像数据，分辨率必须和调用`sp_open_vps`接口是的原始帧分辨率一致。

**【参数】**

- `obj`： 已经初始化的`VIO`对象指针
- `image_buffer`：需要处理的图像帧数据，必须是 `NV12` 格式的图像数据，分辨率必须和调用`sp_open_vps`接口是的原始帧分辨率一致。
- `size`: 帧大小

**【返回类型】**  

成功返回 0，失败返回 -1

## ENCODER（编码模块）API

`ENCODER` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_init_encoder_module | **初始化编码模块对象** |
| sp_release_encoder_module | **销毁编码模块对象** |
| sp_start_encode | **创建图像编码通道** |
| sp_stop_encode | **关闭图像编码通道** |
| sp_encoder_set_frame | **向编码通道传入图像帧** |
| sp_encoder_get_stream | **从编码通道获取编码好的码流** |

### sp_init_encoder_module  

**【函数原型】**  

`void *sp_init_encoder_module()`

**【功能描述】**  

初始化编码模块对象，在使用编码模块时需要调用获得操作句柄。

**【参数】**

无

**【返回类型】**  

成功返回一个`ENCODER`对象指针，失败返回`NULL`。

### sp_release_encoder_module  

**【函数原型】**  

`void sp_release_encoder_module(void *obj)`

**【功能描述】**  

销毁编码模块对象。

**【参数】**

 - `obj`: 调用初始化接口时得到的对象指针。

**【返回类型】**  

无

### sp_start_encode  

**【函数原型】**  

`int32_t sp_start_encode(void *obj, int32_t chn, int32_t type, int32_t width, int32_t height, int32_t bits)`

**【功能描述】**  

创建一路图像编码通道，支持最多创建 `32` 路编码，编码类型支持 `H264`, `H265` 和 `MJPEG`。

**【参数】**

- `obj`： 已经初始化的`ENCODER`对象指针
- `chn`：需要创建的编码通道号，支持 0 ~ 31
- `type`：图像编码类型，支持 `SP_ENCODER_H264`，`SP_ENCODER_H265` 和 `SP_ENCODER_MJPEG`。
- `width`：输入给编码通道的图像数据分辨率-宽
- `height`：输入给编码通道的图像数据分辨率-高
- `bits`：编码码率，常用值为 512, 1024, 2048, 4096, 8192, 16384 等码率（单位 Mbps），其他值也可以，码率也大编码的图像越清晰，压缩率越小，码流数据越大。

**【返回类型】**  

成功返回 0，失败返回 -1

### sp_stop_encode  

**【函数原型】**  

`int32_t sp_stop_encode(void *obj)`

**【功能描述】**  

关闭打开的编码通道。

**【参数】**

- `obj`： 已经初始化的`ENCODER`对象指针

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_encoder_set_frame  

**【函数原型】**  

`int32_t sp_encoder_set_frame(void *obj, char *frame_buffer, int32_t size)`

**【功能描述】**  

向编码通道传入需要编码的图像帧数据，格式必须为 `NV12`。

**【参数】**

- `obj`： 已经初始化的`ENCODER`对象指针
- `frame_buffer`：需要编码的图像帧数据，必须是 `NV12` 格式，分辨率必须和调用`sp_start_encode`接口是的图像帧分辨率一致。
- `size`：图像帧数据大小，`NV12` 格式的图像的大小计算公式为 width * height * 3 / 2。

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_encoder_get_stream  

**【函数原型】**  

`int32_t sp_encoder_get_stream(void *obj, char *stream_buffer)`

**【功能描述】**  

从编码通道获取编码好的码流数据。

**【参数】**

- `obj`： 已经初始化的`ENCODER`对象指针
- `stream_buffer`：获取成功后，码流数据会存在本buffer中。此buffer的大小需要根据编码分辨率和码率进行调整。

**【返回类型】** 

成功返回码流数据的size，失败返回 -1

## DECODER（解码模块）API

`DECODER` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_init_decoder_module | **初始化解码模块对象** |
| sp_release_decoder_module | **销毁解码模块对象** |
| sp_start_decode | **创建图像解码通道** |
| sp_stop_decode | **关闭图像解码通道** |
| sp_decoder_get_image | **从解码通道获取解码后的图像帧** |
| sp_decoder_set_image | **向解码通道传入需要解码的码流数据** |

### sp_init_decoder_module  

**【函数原型】**  

`void *sp_init_decoder_module()`

**【功能描述】**  

初始化解码模块对象，在使用解码模块时需要调用获得操作句柄，支持H264、H265和Mjpeg格式的视频码流。

**【参数】**

无。

**【返回类型】** 

成功返回`DECODER`对象，失败返回 NULL。

### sp_release_decoder_module  

**【函数原型】**  

`void sp_release_decoder_module(void *obj)`

**【功能描述】**  

销毁解码模块对象。

**【参数】**

 - `obj`: 调用初始化接口时得到的对象指针。

**【返回类型】**  

无

### sp_start_decode  

**【函数原型】**  

`int32_t sp_start_decode(void *obj, const char *stream_file, int32_t video_chn, int32_t type, int32_t width, int32_t height)`

**【功能描述】**  

创建一个解码通道，设置通道号、解码的码流类型、图像帧分辨率。

**【参数】**

- `obj`： 已经初始化的`DECODER`对象指针
- `stream_file`：当 `stream_file` 设置为一个码流文件名时，表示对这个码流文件进行解码，例如设置H264的码流文件“stream.h264”, 当 `stream_file` 传入空字符串时，表示解码的数据流需要通过调用 `sp_decoder_set_image` 传入。
- `video_chn`：解码通道号，支持 0-31。
- `type`：解码的数据类型，支持 `SP_ENCODER_H264`，`SP_ENCODER_H265` 和 `SP_ENCODER_MJPEG`。
- `width`：解码出来的图像帧的分辨率 - 宽
- `height`：解码出来的图像帧的分辨率 - 高

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_stop_decode  

**【函数原型】**  

`int32_t sp_stop_decode(void *obj)`

**【功能描述】**  

关闭解码通道。

**【参数】**

- `obj`： 已经初始化的`DECODER`对象指针

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_decoder_get_image  

**【函数原型】**  

`int32_t sp_decoder_get_image(void *obj, char *image_buffer)`

**【功能描述】**  

从解码通道获取解码后的图像帧数据，返回的图像数据格式为 `NV12` 的 `YUV` 图像。

**【参数】**

- `obj`：已经初始化的`DECODER`对象指针
- `image_buffer`：返回的图像帧数据，这个buffer大小与图像分辨率的关系为 width * height * 3 / 2。

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_decoder_set_image  

**【函数原型】**  

`int32_t sp_decoder_set_image(void *obj, char *image_buffer, int32_t chn, int32_t size, int32_t eos)`

**【功能描述】**  

向已经打开的解码通道送入码流数据。
如果是解码 H264 或 H265 码流，需要先发送3-5帧数据，让解码器完成帧缓存后，再获取解码帧数据。
如果解码 H264 码流，首先第一帧送入解码的数据需要是 sps 和 pps 的描述信息，否者解码器会报错退出。

**【参数】**

- `obj`： 已经初始化的`DECODER`对象指针。
- `image_buffer`：码流数据指针。
- `chn`：解码器通道号，需要是调用 `sp_start_decode` 打开过的通道号。
- `size`：码流数据大小。
- `eos`：是否是最后一帧数据。

**【返回类型】** 

成功返回 0，失败返回 -1

## DISPLAY（显示模块）API

`DISPLAY` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_init_display_module | **初始化显示模块对象** |
| sp_release_display_module | **销毁显示模块对象** |
| sp_start_display | **创建视频显示通道** |
| sp_stop_display | **关闭视频显示通道** |
| sp_display_set_image | **向视频显示通道传入图像** |
| sp_display_draw_rect | **在显示通道上绘制矩形框** |
| sp_display_draw_string | **在显示通道上绘制字符串** |
| sp_get_display_resolution | **获取显示器的分辨率** |

### sp_init_display_module  

**【函数原型】**  

`void *sp_init_display_module()`

**【功能描述】**  

初始化显示模块对象，本模块支持把视频图像数据显示到 `HDMI` 接口的显示器上，并且提供在显示画面上绘制矩形框和文字的功能。

**【参数】**

无

**【返回类型】** 

成功返回 `DISPLAY` 对象指针，失败返回 NULL。

### sp_release_display_module  

**【函数原型】**  

`void sp_release_display_module(void *obj)`

**【功能描述】**  

销毁 `DISPLAY` 对象。

**【参数】**

- `obj`： 已经初始化的`DISPLAY`对象指针

**【返回类型】** 

无

### sp_start_display  

**【函数原型】**  

`int32_t sp_start_display(void *obj, int32_t chn, int32_t width, int32_t height)`

**【功能描述】**  

创建一个显示通道，RDK X3开发板支持4个通道，2个视频层，2个图形层。支持的最大分辨率为 `1920 x 1080`, 最大帧率 `60fps`。

**【参数】**

- `obj`： 已经初始化的`DISPLAY`对象指针
- `chn`： 通道号，支持0-3， 如果使用的是桌面系统，0通道用作了图形化系统，所以应用程序请使用通道1。2和3通道一般用来绘制矩形框或者叠加文字信息。
- `width`：显示输出分辨率 - 宽
- `height`：显示输出分辨率 - 高

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_stop_display  

**【函数原型】**  

`int32_t sp_stop_display(void *obj)`

**【功能描述】**  

关闭显示通道。

**【参数】**

- `obj`： 已经初始化的`DISPLAY`对象指针

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_display_set_image  

**【函数原型】**  

`int32_t sp_display_set_image(void *obj, char *addr, int32_t size, int32_t chn)`

**【功能描述】**  

让 `addr` 中的图像数据显示到显示通道 `chn`。 图像格式只支持 `NV12` 的 `YUV` 图像。

**【参数】**

- `obj`：已经初始化的`DISPLAY`对象指针
- `addr`：图像数据，图像格式只支持 `NV12`。
- `size`：图像数据大小，计算公式为： width * height * 3 / 2
- `chn`：显示通道，与 `sp_start_display` 接口使用的通道号对应。

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_display_draw_rect  

**【函数原型】**  

`int32_t sp_display_draw_rect(void *obj, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t chn, int32_t flush, int32_t color, int32_t line_width)`

**【功能描述】**  

在显示模块的图形层绘制矩形框。

**【参数】**

- `obj`： 已经初始化的`DISPLAY`对象指针
- `x0`：绘制矩形框第一个坐标的x值
- `y0`：绘制矩形框第一个坐标的y值
- `x1`：绘制矩形框第二个坐标的x值
- `y1`：绘制矩形框第二个坐标的y值
- `chn`：chn 显示输出层，2~3为图形层
- `flush`：是否清零当前图形层buffer
- `color`：矩形框颜色（颜色格式为ARGB8888）
- `line_width`：矩形框的线宽

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_display_draw_string  

**【函数原型】**  

`int32_t sp_display_draw_string(void *obj, int32_t x, int32_t y, char *str, int32_t chn, int32_t flush, int32_t color, int32_t line_width)`

**【功能描述】**  

在显示模块的图形层绘制矩形框。

**【参数】**

- `obj`： 已经初始化的`DISPLAY`对象指针
- `x`：绘制字符串起始坐标的x值
- `y`：绘制字符串起始坐标的y值
- `str`：需要绘制的字符串（需要是GB2312编码）
- `chn`：chn 显示输出层，2~3为图形层
- `flush`：是否清零当前图形层buffer
- `color`：矩形框颜色（颜色格式为ARGB8888）
- `line_width`：文字的线宽

**【返回类型】** 

成功返回 0，失败返回 -1

### sp_get_display_resolution  

**【函数原型】**  

`void sp_get_display_resolution(int32_t *width, int32_t *height)`

**【功能描述】**  

获取当前接入的显示器分辨率。

**【参数】**

- `width`： 需要获取的分辨率 - 宽
- `height`：需要获取的分辨率 - 高

**【返回类型】** 

无。

## BPU（算法推理模块）API

`BPU` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_init_bpu_module | **初始化算法推理模块对象，创建算法推理任务** |
| sp_bpu_start_predict | **进行AI算法推理，获得推理结果** |
| sp_release_bpu_module | **关闭算法推理任务** |
| sp_init_bpu_tensors | **分配tensor内存** |
| sp_deinit_bpu_tensor | **销毁tensor内存** |


### sp_init_bpu_module  

**【函数原型】**  

`bpu_module *sp_init_bpu_module(const char *model_file_name)`

**【功能描述】**  

打开`model_file_name`算法模型，初始化一个算法推理任务。

**【参数】**

- `model_file_name`： 算法模型文件，需要是经过地平线AI算法工具链转换的或者训练得到的定点模型。

**【返回类型】** 

AI算法推理任务对象。

### sp_bpu_start_predict  

**【函数原型】**  

`int32_t sp_bpu_start_predict(bpu_module *bpu_handle, char *addr)`

**【功能描述】**  

传入图像数据完成AI算法推理，返回算法结果。

**【参数】**

- `bpu_handle`： 算法推理任务对象
- `addr`：图像数据输入

**【返回类型】** 

无。

### sp_init_bpu_tensors 

**【函数原型】**  

` int32_t sp_init_bpu_tensors(bpu_module *bpu_handle, hbDNNTensor *output_tensors)`

**【功能描述】**  

初始化并分配内存给传入的`tensor`。

**【参数】**

- `bpu_handle`： 算法推理任务对象
- `output_tensors`：`tensor`地址

**【返回类型】** 

无。

### sp_deinit_bpu_tensor 

**【函数原型】**  

` int32_t sp_deinit_bpu_tensor(hbDNNTensor *tensor, int32_t len)`

**【功能描述】**  

将传入的`tensor`释放并回收内存。

**【参数】**

- `tensor`： 带出来`tensor`指针
- `output_tensors`：`tensor`地址

**【返回类型】** 

无。


### sp_release_bpu_module  

**【函数原型】**  

`int32_t sp_release_bpu_module(bpu_module *bpu_handle)`

**【功能描述】**  

关闭算法推理任务。

**【参数】**

- `bpu_handle`： 算法推理任务对象

**【返回类型】** 

成功返回 0，失败返回 -1。

## SYS（模块绑定）API

`SYS` API提供了以下的接口：

| 函数 | 功能 |
| ---- | ----- |
| sp_module_bind | **绑定数据源、目标模块** |
| sp_module_unbind | **解除模块间的绑定** |

### sp_module_bind  

**【函数原型】**  

`int32_t sp_module_bind(void *src, int32_t src_type, void *dst, int32_t dst_type)`

**【功能描述】**  

本接口可以把 `VIO`，`ENCODER`，`DECODER`，`DISPLAY`, 这四个模块的输出与输入进行内部绑定，绑定后的两个模块的数据会在内部自动流转，无需用户操作。比如绑定 `VIO` 和 `DISPLAY` 后，打开的mipi摄像头的数据会直接显示到显示屏上，不需要调用`VIO`的`sp_vio_get_frame`接口获取数据，之后再调用`DISPLAY`的`sp_display_set_image`接口进行显示。

支持绑定的模块关系如下：

| 源数据模块 | 目标数据模块 |
| ---- | ----- |
| VIO | ENCODER |
| VIO | DISPLAY |
| DECODER | ENCODER |
| DECODER | DISPLAY |

**【参数】**

- `src`： 数据源模块的对象指针（调用各模块初始化接口得到）
- `src_type`：源数据模块类型，支持 `SP_MTYPE_VIO` 和 `SP_MTYPE_DECODER`
- `dst`： 目标模块的对象指针（调用各模块初始化接口得到）
- `dst_type`：目标数据模块类型，支持 `SP_MTYPE_ENCODER` 和 `SP_MTYPE_DISPLAY`

**【返回类型】**  

成功返回 0，失败返回其他值。

### sp_module_unbind  

**【函数原型】**  

`int32_t sp_module_unbind(void *src, int32_t src_type, void *dst, int32_t dst_type)`

**【功能描述】**  

本接口完成已经绑定的两个模块的解绑，模块退出前需要先完成解绑。

**【参数】**

- `src`： 数据源模块的对象指针（调用各模块初始化接口得到）
- `src_type`：源数据模块类型，支持 `SP_MTYPE_VIO` 和 `SP_MTYPE_DECODER`
- `dst`： 目标模块的对象指针（调用各模块初始化接口得到）
- `dst_type`：目标数据模块类型，支持 `SP_MTYPE_ENCODER` 和 `SP_MTYPE_DISPLAY`

**【返回类型】**  

成功返回 0，失败返回其他值。
