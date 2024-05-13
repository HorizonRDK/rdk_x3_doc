---
sidebar_position: 1
---

# 1.1 准备工作

```mdx-code-block
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
```

在使用RDK X3开发板前，需要做下述准备工作。

## 开发准备

### **供电**

<Tabs groupId="rdk-type">
<TabItem value="x3" label="RDK X3">

RDK X3开发板通过USB Type C接口供电，需要使用支持**5V/3A**的电源适配器为开发板供电，推荐使用[基础配件清单](/hardware_development/rdk_x3/accessory#basic_accessories)中推荐的电源适配器型号。

</TabItem>

<TabItem value="x3md" label="RDK X3 Module">

RDK X3 Module通过载板上的电源接口供电，[官方载板](https://localhost:3000/documents_rdk/hardware_development/rdk_x3_module/interface)通过DC接口供电，推荐使用认证配件清单中推荐的**12V/2A**适配器。

</TabItem>

<TabItem value="ultra" label="RDK Ultra">

RDK Ultra开发板通过DC接口供电，推荐使用[官方套件](https://localhost:3000/documents_rdk/hardware_development/rdk_ultra/interface)中自带的电源适配器，或者使用至少**12V/5A**的电源适配器供电。

</TabItem>
</Tabs>

:::caution

请不要使用电脑USB接口为开发板供电，否则会因供电不足造成开发板**异常断电、反复重启**等异常情况。

更多问题的处理，可以查阅 [常见问题](../category/common_questions) 章节。

:::



### **存储** 
<Tabs groupId="rdk-type">
<TabItem value="x3" label="RDK X3">

RDK X3开发板采用Micro SD存储卡作为系统启动介质，推荐至少8GB容量的存储卡，以便满足Ubuntu系统、应用功能软件对存储空间的需求。

</TabItem>

<TabItem value="x3md" label="RDK X3 Module">

RDK X3 Module板载eMMC（可选），支持从eMMC和SD卡两种模式启动系统。

</TabItem>

<TabItem value="ultra" label="RDK Ultra">

RDK Ultra板载64GB eMMC存储空间，不需要额外准备存储卡。

</TabItem>
</Tabs>



### **显示** 
<Tabs groupId="rdk-type">
<TabItem value="x3" label="RDK X3">

RDK X3开发板支持HDMI显示接口，通过HDMI线缆连接开发板和显示器，支持图形化桌面显示。

</TabItem>

<TabItem value="x3md" label="RDK X3 Module">

RDK X3 Module支持HDMI显示接口，通过HDMI线缆连接官方载板和显示器，支持图形化桌面显示。

</TabItem>

<TabItem value="ultra" label="RDK Ultra">

RDK Ultra开发板支持HDMI显示接口，通过HDMI线缆连接开发板和显示器，支持图形化桌面显示。

</TabItem>
</Tabs>



### **网络连接**
<Tabs groupId="rdk-type">
<TabItem value="x3" label="RDK X3">

RDK X3开发板支持以太网、Wi-Fi两种网络接口，用户可通过任意接口实现网络连接功能。

</TabItem>

<TabItem value="x3md" label="RDK X3 Module">

RDK X3 Module支持以太网、Wi-Fi（可选）两种网络接口，用户可通过任意接口实现网络连接功能。

</TabItem>

<TabItem value="ultra" label="RDK Ultra">

RDK Ultra开发板支持以太网、Wi-Fi两种网络接口，用户可通过任意接口实现网络连接功能。

</TabItem>
</Tabs>



## **常见问题**  

首次使用开发板时的常见问题如下：

- **<font color='Blue'>上电不开机</font>** ：请确保使用[供电](#供电)中推荐的适配器供电；请确保开发板的Micro SD卡或eMMC已经烧录过Ubuntu系统镜像
- **<font color='Blue'>USB Host接口无反应</font>** ：请确保开发板Micro USB接口没有接入数据线
- **<font color='Blue'>使用中热插拔存储卡</font>** ：开发板不支持热插拔Micro SD存储卡，如发生误操作请重启开发板



## **注意事项**

- 禁止带电时拔插除USB、HDMI、网线之外的任何设备
- RDK X3的 Type C USB接口仅用作供电 
- 选用正规品牌的USB Type C 口供电线，否则会出现供电异常，导致系统异常断电的问题



:::tip

更多问题的处理，可以查阅 [常见问题](../category/common_questions) 章节，同时可以访问 [地平线开发者官方论坛](https://developer.horizon.ai/forum) 获得帮助。

:::