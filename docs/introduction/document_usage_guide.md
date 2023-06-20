---
sidebar_position: 2
---

# 1.2 文档使用指引

地平线RDK套件用户手册，是原旭日X3派用户手册的2.0升级版本。本文档基于RDK X3 开发板使用的2.0版本操作系统编写，为开发者提供关于RDK X3(X3派、X3模组)产品的使用说明和开发指导，内容涵盖硬件设计、系统定制、应用开发、算法工具链等多个方面。欢迎用户更新体验，具体方法请参考 [快速开始](../category/getting_start)。


:::caution
对于仍需使用X3派1.0版本系统的用户，可从下述链接中获取资料：

[旭日X3派用户手册](https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html)<br/>
[旭日X3派Ubuntu镜像](https://pan.horizon.ai/index.php/s/xtGGeQ25HEFXXWb)<br/>
[旭日X3派资料包](https://developer.horizon.ai/api/v1/static/fileData/X3%E6%B4%BE%E8%B5%84%E6%96%99%E5%8C%85_20220711175326.zip)<br/>

用户如需确认系统版本号，可通过该命令查询 cat /etc/version
:::

下面将对本文的整体内容划分进行介绍，帮助用户快速了解文档的结构和内容，以便更好地利用文档进行开发工作。

[快速开始](../category/getting_start)  
    介绍系统安装，示例使用的入门说明，帮助用户快速上手使用开发板。  

[系统配置](../category/configuration)  
    介绍一系列配置步骤和技巧，以确保系统能够正常工作并满足特定的需求，引导用户进行系统的配置，包括系统升级、网络、蓝牙的配置。  

[第一个应用程序](../category/first_application)  
    介绍系统中预装的功能示例，如io管脚控制、视频采集，算法推理等。  

[Python开发指南](../category/python_software_development)  
   介绍Python语言版本的视频、图像、算法简易接口的使用方法，此接口简单易用，方便用户快速上手，基于更底层的多媒体接口进行了封装。  

[C/C++开发指南](../category/clang_software_development)  
    介绍C/C++语言版本视频、图像、算法简易接口和libdnn算法接口库的使用方法，本章节还提供了C/C++在RDK X3开发板上的应用示例，帮助用户更快速的开发。  

[Linux开发指南](../category/system_software_development)  
   介绍操作系统软件开发的相关内容，包括开发环境的安装和配置、平台Ubuntu系统的编译和构建方法、驱动程序的开发、系统调试和优化等方面的指引。  

[多媒体开发指南](../category/multimedia_software_development)  
    介绍了视频、图像、多媒体底层接口的使用方法，涵盖了图像处理、音频处理、视频处理、视频编解码等方面的技术和示例，接口功能丰富，可以实现复杂、灵活的功能需求。

[硬件开发指南](../category/hardware)  
   介绍了RDK X3（旭日X3派）、RDK X3 Module（旭日X3模组）硬件规格接口、设计文件及设计指导，提供规格书、原理图、尺寸图等设计资料。

[算法工具链开发指南](../category/quant_toolchain_development)  
   介绍地平线算法量化工具链的使用方法，涵盖了常用的算法模型、开发工具的使用和优化技巧等内容。  

[常见问题](../category/common_questions)  
    本章节回答了用户在使用开发者套件过程中可能遇到的常见问题和疑惑。它提供了解决方案和技巧，帮助用户解决常见问题并顺利进行开发工作。