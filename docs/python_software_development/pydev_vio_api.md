---
sidebar_position: 5
---

# 5.5 多媒体接口说明

## 5.5.1 概要介绍

开发板Ubuntu系统预装了Python版本的`libsrcampy`图像多媒体模块，可以创建`Camera`，`Encode`，`Decode`，`Display`等几种对象，用于完成摄像头图像采集、图像处理、视频编码、视频解码和显示输出等功能。

模块基础使用方式如下：

```python
from hobot_vio import libsrcampy

#create camera object
camera = libsrcampy.Camera()

#create encode object
encode = libsrcampy.Encode()

#create decode object
decode = libsrcampy.Decode()

#create display object
display = libsrcampy.Display()
```

## 5.5.2 Camera对象{#camera}

Camera对象用于完成MIPI Camera的图像采集和处理功能，包含了`open_cam`、`open_vps`、`get_img`、`set_img`、`close_cam`等几种方法，详细说明如下：

### 5.5.2.1 open_cam

<font color='Blue'>【功能描述】</font>  

打开指定通道的MIPI摄像头，并设置摄像头输出帧率、分辨率格式。

<font color='Blue'>【函数声明】</font>  

```python
Camera.open_cam(pipe_id, video_index, fps, width, height)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称      | 定义描述                  | 取值范围    |
| ----------- | ------------------------ | --------  |
| pipe_id     | camera对应的pipeline通道号  | 默认从0开始，范围0~7  |
| video_index | camera对应的host编号，-1表示自动探测，编号可以查看 /etc/board_config.json 配置文件 | 取值 -1, 0 , 1, 2 |
| fps         | camera图像输出帧率          | 依据camera型号而定，默认值30   |
| width       | camera图像输出宽度    |  视camera型号而定，默认值1920   |
| height      | camera图像输出高度  |    视camera型号而定，默认值1080 |

<font color='Blue'>【使用方法】</font> 

```python
#create camera object
camera = libsrcampy.Camera()

#open MIPI Camera, fps: 30, solution: 1080p
ret = camera.open_cam(0, -1, 30, 1920, 1080)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font>  
`width`，`height`参数支持`list`类型输入，表示使能camera多组不同分辨率输出。`list`最多支持4组缩小，1组放大，缩放区间为camera原始分辨率的`1/8~1.5`倍之间。使用方式如下：
```python
ret = cam.open_cam(0, -1, 30, [1920, 1280], [1080, 720])
```

<font color='Blue'>【参考代码】</font>  

无

### 5.5.2.2 open_vps

<font color='Blue'>【功能描述】</font>

使能指定camera通道的vps(video process)图像处理功能，支持对输入图像完成缩放、旋转、裁剪等功能。

<font color='Blue'>【函数声明】</font>  

```python
Camera.open_vps(pipe_id, proc_mode, src_width, src_height, dst_width, dst_height, crop_rect, rotate, src_size, dst_size)
```

<font color='Blue'>【参数描述】</font>  


| 参数名称      | 定义描述                  | 取值范围    |
| ----------- | ------------------------ | --------  |
| pipe_id    | camera对应的pipeline通道号  | 默认从0开始，范围0~7  |
| proc_mode  | 图像处理模式配置，支持缩放、旋转、裁剪   | 范围1~4，分别表示`缩放`、`缩放+裁剪`、`缩放+旋转`、`缩放+裁剪+旋转`|
| src_width  | 图像输入宽度                 | 视camera输出宽度而定 |
| src_height | 图像输入高度                 | 视camera输出高度而定 |
| dst_width  | 图像输出宽度 | 输入宽度的`1/8~1.5`倍 |
| dst_height | 图像输出高度 | 输入高度的`1/8~1.5`倍 |
| crop_rect  | 裁剪区域的宽高，输入格式[x, y] | 不超过输入图像尺寸 |
| rotate     | 旋转角度，最多支持两个通道旋转 | 范围0~3，分别表示`不旋转`、`90度` `180度`、`270度` |
|    src_size | 保留参数 | 默认不需要配置 |
|    dst_size | 保留参数 | 默认不需要配置 |

<font color='Blue'>【使用方法】</font> 

```python
#create camera object
camera = libsrcampy.Camera()

#enable vps function
ret = camera.open_vps(1, 1, 1920, 1080, 512, 512)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 
- vps处理功能最多支持6个通道输出，5路缩小，1路放大，缩放区间为原始分辨率的`1/8~1.5`倍之间，多通道配置通过输入参数`list`传递。
- 图像裁剪功能以图像左上角为原点，按照配置尺寸进行裁剪
- 图像裁剪会在缩放、旋转操作之前进行，多通道配置通过输入参数`list`传递。

```python
#creat camera object
camera = libsrcampy.Camera()

#enable vps function
#input: 4k, output0: 1080p, output1: 720p
#ouput0 croped by [2560, 1440]
ret = camera.open_vps(0, 1, 3840, 2160, [1920, 1280], [1080, 720], [2560, 1440])
```

<font color='Blue'>【参考代码】</font>  
无

### 5.5.2.3 get_img

<font color='Blue'>【功能描述】</font>

获取camera对象的图像输出，需要在`open_cam`、`open_vps`之后调用

<font color='Blue'>【函数声明】</font> 

```python
Camera.get_img(module, width, height)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称 | 定义描述                 | 取值范围     |
| -------- | ------- | ----------- |
| module   | 需要获取图像的模块 | 默认为2 |
| width    | 需要获取图像的宽度 | `open_cam`、`open_vps`设置的输出宽度 |
| height   | 需要获取图像的高度 | `open_cam`、`open_vps`设置的输出高度 |


<font color='Blue'>【使用方法】</font> 

```python
cam = libsrcampy.Camera()

#open MIPI Camera, fps: 30, solution: 1080p
ret = cam.open_cam(0, 1, 30, 1920, 1080)

#wait for 1s
time.sleep(1)

#get one image from camera
img = cam.get_img(2)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败   |

<font color='Blue'>【注意事项】</font> 

该方法需要在`open_cam`、`open_vps`之后调用  

<font color='Blue'>【参考代码】</font>  

```python
import sys, os, time

from hobot_vio import libsrcampy

def test_camera():
    cam = libsrcampy.Camera()

    #open MIPI camera, fps: 30, solution: 1080p
    ret = cam.open_cam(0, 1, 30, 1920, 1080)
    print("Camera open_cam return:%d" % ret)

    # wait for 1s
    time.sleep(1)

    #get one image from camera   
    img = cam.get_img(2)
    if img is not None:
        #save file
        fo = open("output.img", "wb")
        fo.write(img)
        fo.close()
        print("camera save img file success")
    else:
        print("camera save img file failed")
    
    #close MIPI camera
    cam.close_cam()
    print("test_camera done!!!")

test_camera()
```

### 5.5.2.4 set_img

<font color='Blue'>【功能描述】</font>

向`vps`模块输入图像，并触发图像处理操作

<font color='Blue'>【函数声明】</font>  

```python
Camera.set_img(img)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称 | 定义描述     | 取值范围      |
| -------- | -------------------- | ----- |
| img      | 需要处理的图像数据 | 跟vps输入尺寸保持一致 |

<font color='Blue'>【使用方法】</font> 

```python
camera = libsrcampy.Camera()

#enable vps function, input: 1080p, output: 512x512
ret = camera.open_vps(1, 1, 1920, 1080, 512, 512)
print("Camera vps return:%d" % ret)

fin = open("output.img", "rb")
img = fin.read()
fin.close()

#send image to vps module
ret = vps.set_img(img)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

该接口需要在`open_vps`之后调用

<font color='Blue'>【参考代码】</font>  

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def test_camera_vps():
    vps = libsrcampy.Camera()

    #enable vps function, input: 1080p, output: 512x512
    ret = vps.open_vps(1, 1, 1920, 1080, 512, 512)
    print("Camera vps return:%d" % ret)

    fin = open("output.img", "rb")
    img = fin.read()
    fin.close()

    #send image data to vps
    ret = vps.set_img(img)
    print ("Process set_img return:%d" % ret)

    fo = open("output_vps.img", "wb+")

    #get image data from vps
    img = vps.get_img()
    if img is not None:
        fo.write(img)
        print("encode write image success")
    else:
        print("encode write image failed")
    fo.close()

    #close vps function
    vps.close_cam()
    print("test_camera_vps done!!!")

test_camera_vps():
```

### 5.5.2.5 close_cam

<font color='Blue'>【功能描述】</font>

关闭使能的MIPI camera摄像头

<font color='Blue'>【函数声明】</font>  

```python
Camera.close_cam()
```

<font color='Blue'>【参数描述】</font>  

无

<font color='Blue'>【使用方法】</font> 

```python
cam = libsrcampy.Camera()

#open MIPI camera, fps: 30, solution: 1080p
ret = cam.open_cam(0, 1, 30, 1920, 1080)
print("Camera open_cam return:%d" % ret)

#close MIPI camera
cam.close_cam()
```

<font color='Blue'>【返回值】</font>  

无

<font color='Blue'>【注意事项】</font> 

无

<font color='Blue'>【参考代码】</font>  

无

## 5.5.3 Encoder对象
Encoder对象实现了对视频数据的编码压缩功能，包含了`encode`、`encode_file`、`get_img`、`close`等几种方法，详细说明如下：

### 5.5.3.1 encode

<font color='Blue'>【功能描述】</font>

配置并使能encode编码模块

<font color='Blue'>【函数声明】</font>

```python
Encoder.encode(video_chn, encode_type , width, height, bits)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称  | 描述           | 取值范围                    |
| --------- | --------------- | ------------------- |
| video_chn | 指定视频编码器的通道号   | 范围0~31 |
| encode_type    | 视频编码类型  | 范围1~3，分别对应`H264`、`H265`、`MJPEG` |
| width     | 输入编码模块的图像宽度      | 不超过4096              |
| height    | 输入编码模块的图像高度      | 不超过4096              |
| bits      | 编码模块的比特率         |    默认8000kbps         |

<font color='Blue'>【使用方法】</font>

```python
#create encode object
encode = libsrcampy.Encoder()

#enable encode channel 0, solution: 1080p, format: H264
ret = encode.encode(0, 1, 1920, 1080)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败   |

<font color='Blue'>【注意事项】</font>

无

<font color='Blue'>【参考代码】</font>

无

### 5.5.3.2 encode_file

<font color='Blue'>【功能描述】</font>

向使能的编码通道输入图像文件，按预定格式进行编码

<font color='Blue'>【函数声明】</font> 

```python
Encoder.encode_file(img)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称 | 描述              | 取值范围                     |
| -------- | ----------------- | --------------------- |
| img      | 需要编码的图像数据，需要使用NV12格式 | 无 |

<font color='Blue'>【使用方法】</font> 

```python
fin = open("output.img", "rb")
input_img = fin.read()
fin.close()

#input image data to encode
ret = encode.encode_file(input_img)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败   |

<font color='Blue'>【注意事项】</font> 

无

<font color='Blue'>【参考代码】</font>  

无

### 5.5.3.3 get_img

<font color='Blue'>【功能描述】</font>

获取编码后的数据

<font color='Blue'>【函数声明】</font>  

```python
Encoder.get_img()
```

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【参数描述】</font>  

无

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |                 
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败   |

<font color='Blue'>【注意事项】</font> 

该接口需要在调用`Encoder.encode()`创建编码通道后使用

<font color='Blue'>【参考代码】</font>  

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def test_encode():
    #create encode object
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, 1920, 1080)
    print("Encoder encode return:%d" % ret)

    #save encoded data to file
    fo = open("encode.h264", "wb+")
    a = 0
    fin = open("output.img", "rb")
    input_img = fin.read()
    fin.close()
    while a < 100:
        #send image data to encoder
        ret = enc.encode_file(input_img)
        print("Encoder encode_file return:%d" % ret)
        #get encoded data
        img = enc.get_img()
        if img is not None:
            fo.write(img)
            print("encode write image success count: %d" % a)
        else:
            print("encode write image failed count: %d" % a)
        a = a + 1

    enc.close()
    print("test_encode done!!!")

test_encode()
```

### 5.5.3.4 close

<font color='Blue'>【功能描述】</font>

关闭使能的编码通道。

<font color='Blue'>【函数声明】</font>  

```python
Encoder.close()
```

<font color='Blue'>【参数描述】</font>  

无

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【返回值】</font>  

| 返回值 | 定义描述 |
| ------ | ----- |
| 0      | 成功  |
| -1    | 失败   |

<font color='Blue'>【注意事项】</font> 

该接口需要在调用`Encoder.encode()`创建编码通道后使用

<font color='Blue'>【参考代码】</font>  

无

## 5.5.4 Decoder对象

Decoder对象实现了对视频数据的解码功能，包含了`decode`、`set_img`、`get_img`、`close`等几种方法，详细说明如下：

### 5.5.4.1 decode

<font color='Blue'>【功能描述】</font>

使能decode解码模块，并对视频文件进行解码

<font color='Blue'>【函数声明】</font>  

```python
Decoder.decode(file, video_chn, decode_type, width, height, dec_mode)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称  | 描述           | 取值范围                    |
| --------- | --------------- | ------------------- |
| file      | 需要解码的文件名     |       无       |
| video_chn | 指定视频解码器的通道号   | 范围0~31 |
| decode_type | 视频解码类型  | 范围1~3，分别对应`H264`、`H265`、`MJPEG` |
| width     | 输入解码模块的图像宽度      | 不超过4096              |
| height    | 输入解码模块的图像高度      | 不超过4096              |

<font color='Blue'>【使用方法】</font> 

```python
#create decode object
decode = libsrcampy.Decoder()

#enable decode channel 0, solution: 1080p, format: H264
ret = dec.decode("encode.h264", 0, 1, 1920, 1080)
```

 <font color='Blue'>【返回值】</font>  

返回值为2个成员的`list`数据

| 返回值                | 定义描述      |
| ---------------- | ----------- |
| list[0] | 0：解码成功，-1：解码失败      | 
| list[1] | 输入码流文件的帧数，解码成功时有效     |

<font color='Blue'>【注意事项】</font> 

无

<font color='Blue'>【参考代码】</font>  

无

### 5.5.4.2 get_img

<font color='Blue'>【功能描述】</font>

获取解码模块的输出结果

<font color='Blue'>【函数声明】</font>
```python
Decoder.get_img()
```

<font color='Blue'>【参数描述】</font>

无

<font color='Blue'>【使用方法】</font>

```python
ret = dec.decode("encode.h264", 0, 1, 1920, 1080)
print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

img = dec.get_img()
```

<font color='Blue'>【返回值】</font>

| 返回值 | 定义描述 |
| ------ | ----- |
| -1      | 解码数据  |

<font color='Blue'>【注意事项】</font>

该接口需要在调用`Decoder.decode()`创建解码通道后使用

<font color='Blue'>【参考代码】</font>

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def test_decode():
    #create decode object
    dec = libsrcampy.Decoder()

    #enable decode function
    #decode input: encode.h264, solution: 1080p, format: h264
    ret = dec.decode("encode.h264", 0, 1, 1920, 1080)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))
    
    #get decoder output
    img = dec.get_img()
    if img is not None:
        #save file
        fo = open("output.img", "wb")
        fo.write(img)
        fo.close()
        print("decode save img file success")
    else:
        print("decode save img file failed")

    dec.close()
    print("test_decode done!!!")

test_decode()
```

### 5.5.4.3 set_img

<font color='Blue'>【功能描述】</font>

将单帧编码数据送入解码模块，并进行解码

<font color='Blue'>【函数声明】</font>  

```python
Decoder.set_img(img, chn, eos)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称 | 定义描述         | 取值范围 |
| -------- | ------------- | --- | 
| img      | 需要解码的单帧数据 | 无 |
| chn      | 解码器通道号      | 范围0~31 |
| eos      | 解码数据是否结束   | 0：未结束，1：结束 |

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

该接口需要在调用`Decoder.decode()`创建解码通道后使用，且解码通道创建时入参`file`置空

<font color='Blue'>【参考代码】</font>  

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def test_cam_bind_encode_decode_bind_display():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
    print("Camera open_cam return:%d" % ret)

    #enable encoder
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, 1920, 1080)
    print("Encoder encode return:%d" % ret)

    #enable decoder
    dec = libsrcampy.Decoder()
    ret = dec.decode("", 0, 1, 1920, 1080)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    ret = libsrcampy.bind(cam, enc)
    print("libsrcampy bind return:%d" % ret)

    a = 0
    while a < 100:
        #get encode image from encoder
        img = enc.get_img()
        if img is not None:
            #send encode image to decoder
            dec.set_img(img)
            print("encode get image success count: %d" % a)
        else:
            print("encode get image failed count: %d" % a)
        a = a + 1

    ret = libsrcampy.unbind(cam, enc)
    dec.close()
    enc.close()
    cam.close_cam()
    print("test_cam_bind_encode_decode_bind_display done!!!")

    test_cam_bind_encode_decode()
```

### 5.5.4.4 close

<font color='Blue'>【功能描述】</font>

关闭解码模块

<font color='Blue'>【函数声明】</font>
```python
Decoder.close()
```

<font color='Blue'>【参数描述】</font>

无

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【返回值】</font>

| 返回值 | 定义描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font>

退出程序时需要调用`close`接口以释放资源。

<font color='Blue'>【参考代码】</font>

无

## 5.5.5 Display对象{#display}

Display对象实现了视频显示功能，可以将图像数据通过`HDMI`接口输出到显示器，该对象包含`display`、`set_img`、`set_graph_rect`、`set_graph_word`、`close`等方法，详细说明如下：

### 5.5.5.1 display
<font color='Blue'>【功能描述】</font>

显示模块初始化，并配置显示参数

<font color='Blue'>【函数声明】</font>  

```python
Display.display(chn, width, height, vot_intf, vot_out_mode)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称     | 定义描述                  | 取值范围      |
| ------------ | ----------------------- | ----------------- |
| chn          | 显示输出层        | 0: 视频层，2: 图形层  |
| width        | 输入图像的宽度       | 不超过1920 |
| height       | 输入图像的高度       | 不超过1080 |
| vot_intf     | 视频接口输出分辨率 | 默认为0，1080p |
| vot_out_mode | 视频输出接口     | 默认为1，HDMI输出 |

<font color='Blue'>【使用方法】</font> 

```python
#create display object
disp = libsrcampy.Display()

#enable display function, solution: 1080p, interface: HDMI
ret = disp.display(0, 1920, 1080, 0, 1)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

开发板HDMI接口分辨率基于显示器EDID获取，目前只支持`1920x1080`、`1280x720`、`1024x600`、`800x480`几种分辨率。使能显示模块时，需要注意配置分辨率跟显示器实际分辨率相匹配。

<font color='Blue'>【参考代码】</font>  

无

### 5.5.5.2 set_img

<font color='Blue'>【功能描述】</font>

向display模块输入显示数据，格式需要为`NV12`

<font color='Blue'>【函数声明】</font>  

```python
Display.set_img(img)
```

<font color='Blue'>【参数描述】</font>  

| 参数名称     | 定义描述                  | 取值范围      |
| ------------ | ----------------------- | ----------------- |
| img          | 需要显示的图像数据        | NV12格式  |

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

该接口需要在使用`display`接口使能显示功能后使用，送入数据需要为`NV12`格式

<font color='Blue'>【参考代码】</font>  

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def test_display():
    #create display object
    disp = libsrcampy.Display()

    #enable display function
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)

    fo = open("output.img", "rb")
    img = fo.read()
    fo.close()

    #send image data to display
    ret = disp.set_img(img)
    print ("Display set_img return:%d" % ret)

    time.sleep(3)

    disp.close()
    print("test_display done!!!")

test_display()
```

### 5.5.5.3 set_graph_rect

<font color='Blue'>【功能描述】</font>

在显示模块的图形层绘制矩形框

<font color='Blue'>【函数声明】</font>

```python
Display.set_graph_rect(x0, y0, x1, y1, chn, flush, color, line_width)
```

<font color='Blue'>【参数描述】</font>

| 参数名称   | 定义描述             |    取值范围            |
| ---------- | ----------------------- | --------- |
| x0         | 绘制矩形框左上角的坐标值x   | 不超过视频画面尺寸   |
| y0         | 绘制矩形框左上角的坐标值y   | 不超过视频画面尺寸   |
| x1         | 绘制矩形框右下角的坐标值x   | 不超过视频画面尺寸   |
| y1         | 绘制矩形框右下角的坐标值y   | 不超过视频画面尺寸   |
| chn        | 图形层通道号 |  范围2~3，默认为2     |
| flush      | 是否清零图形层buffer   | 0：否，1：是      |
| color      | 矩形框颜色设置 |  ARGB8888格式 |
| line_width | 矩形框边的宽度        | 范围1~16，默认为4      |

<font color='Blue'>【使用方法】</font>

```python
#enable graph layer 2
ret = disp.display(2)
print ("Display display 2 return:%d" % ret)

#set osd rectangle
ret = disp.set_graph_rect(100, 100, 1920, 200, chn = 2, flush = 1,  color = 0xffff00ff)
```

<font color='Blue'>【返回值】</font>

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font>

该接口需要在使用`display`接口使能显示功能后使用

<font color='Blue'>【参考代码】</font>

无

### 5.5.5.4 set_graph_word

<font color='Blue'>【功能描述】</font>

在显示模块的图形层绘制字符

<font color='Blue'>【函数声明】</font>

```python
Display.set_graph_word(x, y, str, chn, flush, color, line_width)
```

<font color='Blue'>【参数描述】</font>

| 参数名称   | 描述                    | 取值范围         |
| ---------- | ---------------------- | ------------- |
| x          | 绘制字符的起始坐标值x     | 不超过视频画面尺寸   |
| y          | 绘制字符的起始坐标值y   | 不超过视频画面尺寸   |
| str        | 需要绘制的字符数据 | GB2312编码 |
| chn        | 图形层通道号 |  范围2~3，默认为2     |
| flush      | 是否清零图形层buffer   | 0：否，1：是      |
| color      | 字符颜色设置 |  ARGB8888格式 |
| line_width | 字符线条的宽度        | 范围1~16，默认为1      |

<font color='Blue'>【使用方法】</font>

```python
#enable graph layer 2
ret = disp.display(2)
print ("Display display 2 return:%d" % ret)

#set osd string
string = "horizon"
ret = disp.set_graph_word(300, 300, string.encode('gb2312'), 2, 0, 0xff00ffff)
print ("Display set_graph_word return:%d" % ret)
```

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

该接口需要在使用`display`接口使能显示功能后使用

<font color='Blue'>【参考代码】</font>  

无

### 5.5.5.5 close

<font color='Blue'>【功能描述】</font>

关闭显示模块

<font color='Blue'>【函数声明】</font>  

```python
Display.close()
```

<font color='Blue'>【参数描述】</font>  

无

<font color='Blue'>【使用方法】</font> 

无

<font color='Blue'>【返回值】</font>  

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font> 

该接口需要在使用`display`接口使能显示功能后使用

<font color='Blue'>【参考代码】</font>  

无

### 5.5.5.6 bind接口

<font color='Blue'>【功能描述】</font>

该接口可以把`Camera`、`Encoder`、`Decoder`、`Display`模块的输出与输入数据流进行绑定，绑定后无需用户操作，数据可在绑定模块之间自动流转。例如，绑定 `Camera` 和 `Display` 后，摄像头数据会自动通过显示模块输出到显示屏上，无需调用额外接口。

<font color='Blue'>【函数声明】</font>
```python
    libsrcampy.bind(src, dst)
```

<font color='Blue'>【参数描述】</font>

| 参数名称 | 描述         | 取值范围 |
| -------- | ------------ | --- |
| src      | 源数据模块   |`Camera`、`Encoder`、`Decoder`模块 |
| dst      | 目标数据模块 |`Camera`、`Encoder`、`Decoder`、`Display`模块|

<font color='Blue'>【使用方法】</font>

```python
#create camera object
cam = libsrcampy.Camera()
ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
print("Camera open_cam return:%d" % ret)

#encode start
enc = libsrcampy.Encoder()
ret = enc.encode(0, 1, 1920, 1080)
print("Encoder encode return:%d" % ret)

#bind, input: cam, output: enc
ret = libsrcampy.bind(cam, enc)
print("libsrcampy bind return:%d" % ret)
```

 <font color='Blue'>【返回值】</font>

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font>

无

<font color='Blue'>【参考代码】</font>

无

### 5.5.5.7 unbind接口

<font color='Blue'>【功能描述】</font>

将两个绑定过的模块解绑

<font color='Blue'>【函数声明】</font>
```python
libsrcampy.unbind(src, dst)
```

<font color='Blue'>【参数描述】</font>

| 参数名称 | 描述         | 取值范围 |
| -------- | ------------ | --- |
| src      | 源数据模块   |`Camera`、`Encoder`、`Decoder`模块 |
| dst      | 目标数据模块 |`Camera`、`Encoder`、`Decoder`、`Display`模块|

<font color='Blue'>【使用方法】</font>

```python
#create camera object
cam = libsrcampy.Camera()
ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
print("Camera open_cam return:%d" % ret)

#encode start
enc = libsrcampy.Encoder()
ret = enc.encode(0, 1, 1920, 1080)
print("Encoder encode return:%d" % ret)

#bind, input: cam, output: enc
ret = libsrcampy.bind(cam, enc)
print("libsrcampy bind return:%d" % ret)

#unbind, input: cam, output: enc
ret = libsrcampy.unbind(cam, enc)
print("libsrcampy unbind return:%d" % ret)
```

 <font color='Blue'>【返回值】</font>

| 返回值 | 描述 |
| ------ | ---- |
| 0      | 成功 |
| -1    | 失败 |

<font color='Blue'>【注意事项】</font>

无

<font color='Blue'>【参考代码】</font>

无

## 5.5.6 接口使用示例代码{#vio_demo_code}

以下示例代码包含多个单元测试用例，覆盖了本章节接口的使用方式，具体如下：

```python
import sys, os, time

import numpy as np
import cv2
from hobot_vio import libsrcampy

def get_nalu_pos(byte_stream):
    size = byte_stream.__len__()
    nals = []
    retnals = []

    startCodePrefixShort = b"\x00\x00\x01"

    pos = 0
    while pos < size:
        is4bytes = False
        retpos = byte_stream.find(startCodePrefixShort, pos)
        if retpos == -1:
            break
        if byte_stream[retpos - 1] == 0:
            retpos -= 1
            is4bytes = True
        if is4bytes:
            pos = retpos + 4
        else:
            pos = retpos + 3
        val = hex(byte_stream[pos])
        val = "{:d}".format(byte_stream[pos], 4)
        val = int(val)
        fb = (val >> 7) & 0x1
        nri = (val >> 5) & 0x3
        type = val & 0x1f
        nals.append((pos, is4bytes, fb, nri, type))
    for i in range(0, len(nals) - 1):
        start = nals[i][0]
        if nals[i + 1][1]:
            end = nals[i + 1][0] - 5
        else:
            end = nals[i + 1][0] - 4
        retnals.append((start, end, nals[i][1], nals[i][2], nals[i][3], nals[i][4]))
    start = nals[-1][0]
    end = byte_stream.__len__() - 1
    retnals.append((start, end, nals[-1][1], nals[-1][2], nals[-1][3], nals[-1][4]))
    return retnals

def get_h264_nalu_type(byte_stream):
    nalu_types = []
    nalu_pos = get_nalu_pos(byte_stream)

    for idx, (start, end, is4bytes, fb, nri, type) in enumerate(nalu_pos):
        # print("NAL#%d: %d, %d, %d, %d, %d" % (idx, start, end, fb, nri, type))
        nalu_types.append(type)
    
    return nalu_types

def test_camera():
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, 1920, 1080)
    print("Camera open_cam return:%d" % ret)
    # wait for isp tuning
    time.sleep(1)
    img = cam.get_img(2)
    if img is not None:
        #save file
        fo = open("output.img", "wb")
        fo.write(img)
        fo.close()
        print("camera save img file success")
    else:
        print("camera save img file failed")
    cam.close_cam()
    print("test_camera done!!!")

def test_camera_vps():
    #vps start
    vps = libsrcampy.Camera()
    ret = vps.open_vps(1, 1, 1920, 1080, 512, 512)
    print("Camera vps return:%d" % ret)

    fin = open("output.img", "rb")
    img = fin.read()
    fin.close()
    ret = vps.set_img(img)
    print ("Process set_img return:%d" % ret)

    fo = open("output_vps.img", "wb+")
    img = vps.get_img()
    if img is not None:
        fo.write(img)
        print("encode write image success")
    else:
        print("encode write image failed")
    fo.close()

    vps.close_cam()
    print("test_camera_vps done!!!")

def test_encode():
    #encode file
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, 1920, 1080)
    print("Encoder encode return:%d" % ret)

    #save file
    fo = open("encode.h264", "wb+")
    a = 0
    fin = open("output.img", "rb")
    input_img = fin.read()
    fin.close()
    while a < 100:
        ret = enc.encode_file(input_img)
        print("Encoder encode_file return:%d" % ret)
        img = enc.get_img()
        if img is not None:
            fo.write(img)
            print("encode write image success count: %d" % a)
        else:
            print("encode write image failed count: %d" % a)
        a = a + 1

    enc.close()
    print("test_encode done!!!")

def test_decode():
    #decode start
    dec = libsrcampy.Decoder()

    ret = dec.decode("encode.h264", 0, 1, 1920, 1080)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    img = dec.get_img()
    if img is not None:
        #save file
        fo = open("output.img", "wb")
        fo.write(img)
        fo.close()
        print("decode save img file success")
    else:
        print("decode save img file failed")

    dec.close()
    print("test_decode done!!!")
    
def test_display():
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)
    ret = disp.display(2)
    print ("Display display 2 return:%d" % ret)
    ret = disp.set_graph_rect(100, 100, 1920, 200, chn = 2, flush = 1,  color = 0xffff00ff)
    print ("Display set_graph_rect return:%d" % ret)
    string = "horizon"
    ret = disp.set_graph_word(300, 300, string.encode('gb2312'), 2, 0, 0xff00ffff)
    print ("Display set_graph_word return:%d" % ret)
    
    fo = open("output.img", "rb")
    img = fo.read()
    fo.close()
    ret = disp.set_img(img)
    print ("Display set_img return:%d" % ret)

    time.sleep(3)

    disp.close()
    print("test_display done!!!")

def test_camera_bind_encode():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
    print("Camera open_cam return:%d" % ret)

    #encode start
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, 1920, 1080)
    print("Encoder encode return:%d" % ret)
    ret = libsrcampy.bind(cam, enc)
    print("libsrcampy bind return:%d" % ret)

    enc1 = libsrcampy.Encoder()
    ret = enc1.encode(1, 1, 1280, 720)
    print("Encoder encode return:%d" % ret)
    ret = libsrcampy.bind(cam, enc1)
    print("libsrcampy bind return:%d" % ret)

    #save file
    fo = open("encode.h264", "wb+")
    fo1 = open("encode1.h264", "wb+")
    a = 0
    while a < 100:
        img = enc.get_img()
        img1 = enc1.get_img()
        if img is not None:
            fo.write(img)
            fo1.write(img1)
            print("encode write image success count: %d" % a)
        else:
            print("encode write image failed count: %d" % a)
        a = a + 1
    fo.close()
    fo1.close()

    print("save encode file success")
    ret = libsrcampy.unbind(cam, enc)
    print("libsrcampy unbind return:%d" % ret)
    ret = libsrcampy.unbind(cam, enc1)
    print("libsrcampy unbind return:%d" % ret)

    enc1.close()
    enc.close()
    cam.close_cam()
    print("test_camera_bind_encode done!!!")

def test_camera_bind_display():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, 1280, 720)
    print("Camera open_cam return:%d" % ret)

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1, chn_width = 1280, chn_height = 720)
    print ("Display display 0 return:%d" % ret)
    ret = disp.display(2, chn_width = 1280, chn_height = 720)
    print ("Display display 2 return:%d" % ret)
    disp.set_graph_rect(100, 100, 1920, 200, chn = 2, flush = 1,  color = 0xffff00ff)
    string = "horizon"
    disp.set_graph_word(300, 300, string.encode('gb2312'), 2, 0, 0xff00ffff)
    ret = libsrcampy.bind(cam, disp)
    print("libsrcampy bind return:%d" % ret)
    
    time.sleep(10)

    ret = libsrcampy.unbind(cam, disp)
    print("libsrcampy unbind return:%d" % ret)
    disp.close()
    cam.close_cam()
    print("test_camera_bind_display done!!!")

def test_decode_bind_display():
    #decode start
    dec = libsrcampy.Decoder()
    ret = dec.decode("encode.h264", 0, 1, 1920, 1080)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    dec1 = libsrcampy.Decoder()
    ret = dec1.decode("encode1.h264", 1, 1, 1280, 720)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)
    ret = disp.display(2)
    print ("Display display 2 return:%d" % ret)
    disp.set_graph_rect(100, 100, 1920, 200, chn = 2, flush = 1,  color = 0xffff00ff)
    string = "horizon"
    disp.set_graph_word(300, 300, string.encode('gb2312'), 2, 0, 0xff00ffff)
    ret = libsrcampy.bind(dec, disp)
    print("libsrcampy bind return:%d" % ret)
    
    time.sleep(5)

    ret = libsrcampy.unbind(dec, disp)
    print("libsrcampy unbind return:%d" % ret)
    disp.close()
    dec1.close()
    dec.close()
    print("test_decode_bind_display done!!!")

def test_cam_bind_encode_decode_bind_display():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
    print("Camera open_cam return:%d" % ret)

    #encode file
    enc = libsrcampy.Encoder()
    ret = enc.encode(0, 1, 1920, 1080)
    print("Encoder encode return:%d" % ret)

    #decode start
    dec = libsrcampy.Decoder()
    ret = dec.decode("", 0, 1, 1920, 1080)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)

    ret = libsrcampy.bind(cam, enc)
    print("libsrcampy bind return:%d" % ret)
    ret = libsrcampy.bind(dec, disp)
    print("libsrcampy bind return:%d" % ret)

    a = 0
    while a < 100:
        img = enc.get_img()
        if img is not None:
            dec.set_img(img)
            print("encode get image success count: %d" % a)
        else:
            print("encode get image failed count: %d" % a)
        a = a + 1

    ret = libsrcampy.unbind(cam, enc)
    ret = libsrcampy.unbind(dec, disp)
    disp.close()
    dec.close()
    enc.close()
    cam.close_cam()
    print("test_cam_bind_encode_decode_bind_display done!!!")

def test_cam_vps_display():
    #camera start
    cam = libsrcampy.Camera()
    ret = cam.open_cam(0, 1, 30, [1920, 1280], [1080, 720])
    print("Camera open_cam return:%d" % ret)

    #vps start
    vps = libsrcampy.Camera()
    ret = vps.open_vps(1, 1, 1920, 1080, 512, 512)
    print("Camera vps return:%d" % ret)

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)

    a = 0
    while a < 100:
        img = cam.get_img()
        if img is not None:
            vps.set_img(img)
            print("camera get image success count: %d" % a)
        else:
            print("camera get image failed count: %d" % a)

        img = vps.get_img(2, 1920, 1080)
        if img is not None:
            disp.set_img(img)
            print("vps get image success count: %d" % a)
        else:
            print("vps get image failed count: %d" % a)
        a = a + 1

    disp.close()
    vps.close_cam()
    cam.close_cam()
    print("test_cam_vps_display done!!!")

def test_rtsp_decode_bind_vps_bind_disp(rtsp_url):
    start_time = time.time()
    image_count = 0
    skip_count = 0
    find_pps_sps = 0

    #rtsp start
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_FORMAT, -1) # get stream
    if not cap.isOpened():
        print("fail to open rtsp: {}".format(rtsp_url))
        return -1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #decode start
    dec = libsrcampy.Decoder()
    ret = dec.decode("", 0, 1, width, height)
    print ("Decoder return:%d frame count: %d" %(ret[0], ret[1]))

    #camera start
    vps = libsrcampy.Camera()
    ret = vps.open_vps(0, 1, width, height, [1920, 512], [1080, 512])
    print("Camera open_cam return:%d" % ret)

    #display start
    disp = libsrcampy.Display()
    ret = disp.display(0, 1920, 1080, 0, 1)
    print ("Display display 0 return:%d" % ret)

    ret = libsrcampy.bind(dec, vps)
    print("libsrcampy bind return:%d" % ret)
    ret = libsrcampy.bind(vps, disp)
    print("libsrcampy bind return:%d" % ret)

    a = 0
    while True:
        ret, stream_frame = cap.read()
        if not ret:
            return
        nalu_types = get_h264_nalu_type(stream_frame.tobytes())

        # 送入解码的第一帧需要是 pps，sps, 否则解码器会报 "FAILED TO DEC_PIC_HDR" 异常而退出
        if (nalu_types[0] in [1, 5]) and find_pps_sps == 0:
            continue

        find_pps_sps = 1
        if stream_frame is not None:
            ret = dec.set_img(stream_frame.tobytes(), 0) # 发送码流, 先解码数帧图像后再获取
            if ret != 0:
                return ret
            if skip_count < 5:
                skip_count += 1
                image_count = 0
                continue

    ret = libsrcampy.unbind(dec, vps)
    ret = libsrcampy.unbind(vps, disp)
    disp.close()
    dec.close()
    vps.close_cam()
    cap.release()
    print("test_rtsp_decode_bind_vps_bind_disp done!!!")


test_camera()
test_camera_vps()
test_encode()
test_decode()
test_display()
test_camera_bind_encode()
test_camera_bind_display()
test_decode_bind_display()
test_cam_bind_encode_decode_bind_display()
test_cam_vps_display()

# rtsp_url = "rtsp://127.0.0.1/3840x2160.264"
# test_rtsp_decode_bind_vps_bind_disp(rtsp_url)
```

