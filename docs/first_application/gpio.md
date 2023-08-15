---
sidebar_position: 1
---

# 3.1 GPIO读写操作示例

<iframe src="//player.bilibili.com/player.html?aid=700903305&bvid=BV1rm4y1E73q&cid=1196557887&page=16" scrolling="no" border="0" frameborder="no" framespacing="0" width="100%" height="500" allowfullscreen="true"> </iframe>

开发板 `/app/40pin_samples/` 目录下，预置了多种40PIN管脚的功能测试代码，包括gpio的输入/输出测试、PWM、I2C、SPI、UART等测试。所有测试程序均使用python语言编写，详细信息可以查阅 [40PIN 功能使用](../python_development/40pin_user_guide/40pin_define.md)。

以`/app/40pin_samples/button_led.py`为例，该程序配置`38`号管脚为输入，配置`36`号管脚配置为输出，并根据`38`号管脚的输入状态来控制`36`号管脚的输出状态。

## 环境准备
使用杜邦线连接`38`号管脚到3.3v or GND，以控制其高低电平。

## 运行方式
执行 `button_led.py` 程序，以启动GPIO读写程序

  ```bash
  sunrise@ubuntu:~$ cd /app/40pin_samples/
  sunrise@ubuntu:/app/40pin_samples$ sudo python3 ./button_led.py
  ```

## 预期效果
通过控制`38`号管脚的高低电平，可以改变 `36`号管脚的输出电平值。

  ```bash
  sunrise@ubuntu:/app/40pin_samples$ sudo python3 ./button_led.py
  Starting demo now! Press CTRL+C to exit
  Outputting 0 to Pin 36
  Outputting 1 to Pin 36
  Outputting 0 to Pin 36
  ```