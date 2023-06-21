---
sidebar_position: 3
---

# 7.3 硬件单元测试
## 概述
本章介绍如何进行X3硬件单元测试，所用到的程序和脚本请从 [点击](http://archive.sunrisepi.tech/downloads/unittest/sample_base_test.tar.gz) 下载。
### 测试程序使用方法
1. 将`sample_base_test.tar.gz`上传到X3板端，建议将其放到`/userdate`或者`/app`目录下。
2. 进入存放`sample_base_test.tar.gz`的目录，解压测试程序`tar -xvf sample_base_test.tar.gz`
### 声明
**本测试方案中提供的程序和脚本，未必是最佳选择，不同硬件设计、不同的外设配置，都可能会导致本测试方法失效，或者导致测试出来的数据与指标值存在较大差异，需要有耐心进行调试验证。**

## 环境可靠性测试（定频）
### 测试方法
1.	进入到`test_tools/01_cpu_bpu_ddr`文件夹下，执行`sh cpubpu.sh &`脚本，对`bpu`，`cpu`进行压力测试，当前配置启动双`bpu`运行到最高99%负载，`cpu` 90%以上负载，内存随机读写
2.	可以在脚本中更改运行时间及负载强度
3.	打开终端记录运行log
### 测试标准
1. 高温：45°、低温：-10°、常温下，程序正常执行，不会出现重启挂死的情况。
2. LOG中没有`fail`、`error`、`timeout`等异常打印。
3. 能稳定运行48小时。
4. 运行过程中需要关注CPU、BPU占比，使CPU占比均接近90%，BPU压力接近90%。

## EMMC相关测试
对于EMMC来讲，主要关心它的稳定性和性能。
### EMMC稳定性测试
#### 测试方法
1. 使用开源工具`iozone`对EMMC进行文件系统读写测试。
2. 进入到`test_tools/02_emmc`文件夹下，执行`sh emmc_stability_test.sh &`脚本对EMMC文件系统读写测试。
#### 测试标准
1. 高温：45°、低温：-10°、常温下，程序正常执行，不会出现重启挂死的情况。
2. LOG中没有`fail`、`error`、`timeout`等异常打印。
3. 能稳定运行48小时。
### EMMC性能测试
#### 测试方法
1. 使用开源工具`iozone`对EMMC文件系统读写速度进行测试。
2. Read上限：172.8MB/s、Write上限：35MB/s。
3. 进入到`test_tools/02_emmc`文件夹下，执行`sh emmc_performance_test.sh &`脚本。
#### 测试标准
1. 常温环境下，程序正常执行，不会出现重启挂死等异常。
2. LOG中无`fail`、`error`、`timeout`等异常打印。
3. 统计实际测试读写速度是否符合性能指标。
4. 稳定运行48小时。

## UART压力测试
### 测试方法
1. 镜像要打开`uart1`和`uart2`节点（双发双收），硬件上将`uart1`的`TX`、`RX`与`uart2`的`RX`、`TX`短接。
2. 执行测试脚本：`sh uart1test.sh &`、`sh uart2test.sh &`。
### 测试标准
1. 高温：45°、低温：-10°、常温下，程序正常执行，不会出现重启挂死的情况。
2. LOG中没有`fail`、`error`、`timeout`等异常打印。
3. 能稳定运行48小时。
### 附录
测试源码以及编译可以参考[UART驱动调试指南](./driver_develop_guide#uart_test)

## SPI压力测试
### 测试方法
1. 进入`test_tools/07_spi_test`目录
2. 测试脚本分成`master`和`salve`两种模式，根据spi驱动配置的模式运行相应模式下的脚本
3. 可以采用两块RDK X3开发板，一块把spi配置成`master`模式，一块配置成`salve`模式,配置流程可以参考[SPI调试指南](./driver_develop_guide#SPI_debug_guide)。先执行master端测试脚本：`sh spitest_master.sh &`后执行salve端测试脚本：`sh spitest_salve.sh &`,两个脚本执行间隔应尽可能短。
### 测试标准
1. 高温：45°、低温：-10°、常温下，程序正常执行，不会出现重启挂死的情况。
2. LOG中没有`fail`、`error`、`timeout`等异常打印。
3. 能稳定运行48小时。

## USB驱动性能测试
### 测试方法
#### 开发板侧
1. 使用`CrystalDiskMark`进行测试（软件包在`09-usb_test`目录下）。
2. 开发板输入下面命令:  
```bash
service adbd stop
cd /tmp
dd if=/dev/zero of=/tmp/700M.img bs=1M count=700
losetup -f /tmp/700M.img
losetup -a 
modprobe g_mass_storage file=/dev/loop0 removable=1
```
#### PC侧
1. PC端会出现新磁盘设备的提醒，将其格式化为FAT32格式。
2. PC打开`CrystalDiskMark`，选择刚挂载的X3设备，点击`All`开始测试，若出现空间不足的提示，则调整测试文件大小。
3. 测试完成之后，前两项`SEQ1M*`表示顺序读写速度，后面两项`RND4K*`表示4k小文件随机读写速度。
![10_usb_benchmark](./image/hardware_unit_test/10_usb_benchmark.png)  


  **图片中的速度仅供参考**
### 测试标准
测试结果取CrystalDiskMark SEQ1MQ8T1读写数据  
USB 2.0 : 读写超过**40**MB/s  
USB 3.0 : 读写超过**370**MB/s  

## 网络性能测试
### 测试说明
使用 `iperf3` 工具进行测试（sdk源码包已自带该工具）。  
`iperf3` 是一个 `TCP`、`UDP` 和 `SCTP` 网络带宽测量工具。是用于主动测量IP网络上可达到的最大带宽的工具。
### 测试方法
首先确定好开发板和PC能互相ping通，才能进行下一步测试。

#### PC侧
PC端做服务端，执行`iperf3 -s -f m` 。

#### 开发板侧
开发板做客户端，执行`iperf3 -c 192.168.1.1 -f m -i 1 -t 60` 进行网络测试。

#### iperf3常用参数
对于服务端，`iperf3`常见配置参数如下:
```bash
-s       表示服务器端；
-p port  定义端口号；
-i sec   设置每次报告之间的时间间隔，单位为秒，如果设置为非零值，就会按照此时间间隔输出测试报告，默认值为零
```
对于客户端，`iperf3` 常见配置参数如下:
```shell
-c ip   表示服务器的IP地址；
-p port 表示服务器的端口号；
-t sec  参数可以指定传输测试的持续时间,Iperf在指定的时间内，重复的发送指定长度的数据包，默认是10秒钟.
-i sec  设置每次报告之间的时间间隔，单位为秒，如果设置为非零值，就会按照此时间间隔输出测试报告，默认值为零；
-w size 设置套接字缓冲区为指定大小，对于TCP方式，此设置为TCP窗口大小，对于UDP方式，此设置为接受UDP数据包的缓冲区大小，限制可以接受数据包的最大值.
--logfile    参数可以将输出的测试结果储存至文件中.
-J  来输出JSON格式测试结果.
-R  反向传输,缺省iperf3使用上传模式：Client负责发送数据，Server负责接收；如果需要测试下载速度，则在Client侧使用-R参数即可.
```

### 测试标准
接收带宽：**870Mbits/sec**  

发送带宽：**940Mbits/sec**

## CPU性能测试
### 测试说明
本测试使用`Coremark`工具进行测试，源码和编译好的软件放在`10-cpu_performace`目录下。
CoreMark是一项基准测试程序，其主要目标是测试处理器核心性能，CoreMark标准的测试方法就是在某配置参数组合下单位时间内跑了多少次CoreMark程序，
业界的分数呈现为 `Coremark` / `CPU clock Mhz` / `Core num`，即 `coremark每秒跑的次数` / `cpu时钟频率` / `cpu的核数`，最终得到一个评分。
### 测试方法
1. 解压`coremark-main.zip`，并进入`coremark-main`文件夹
2. 执行`./coremark_single 0x0 0x0 0x66 0 7 1 2000 > ./run1.log`，等待程序执行完毕;执行`./coremark_multi 0x0 0x0 0x66 0 7 1 2000 > ./run2.log`，等待程序执行完毕。
3. 查看`run1.log`里面的**单核**测试成绩，参考如下：
```yaml
2K performance run parameters for coremark.
CoreMark Size    : 666
Total ticks      : 20830
Total time (secs): 20.830000
Iterations/Sec   : 5280.844935
Iterations       : 110000
Compiler version : GCC6.5.0
Compiler flags   :  -O3 -funroll-all-loops -static --param max-inline-insns-auto=550 -DPERFORMANCE_RUN=1  -lrt
Memory location  : Please put data memory location here
                        (e.g. code in flash, data on heap etc)
seedcrc          : 0xe9f5
[0]crclist       : 0xe714
[0]crcmatrix     : 0x1fd7
[0]crcstate      : 0x8e3a
[0]crcfinal      : 0x33ff
Correct operation validated. See README.md for run and reporting rules.
CoreMark 1.0 : 5280.844935 / GCC6.5.0  -O3 -funroll-all-loops -static --param max-inline-insns-auto=550 -DPERFORMANCE_RUN=1  -lrt / Heap
```
注意到`Iterations/Sec`这栏，表示每秒钟迭代多少次，也就是我们上面公式的`coremark`分数。  
根据公式，这颗x3的单核分数为`5280.844935`/`1200`（默认频率）/`1` = `4.400`。属于正常范围。  

`./run2.log`里面保存着**多核心**的成绩，计算多核分数和单核分数类似，此处不再赘述。  

### 测试指标
- 单核分数 X > 4.2
- 四核分数 X > 4.2

### 附录
交叉编译`coremark`流程如下：
1. 进入`coremark-main`目录，将`aarch64/core_portme.mak`中的`CC`编译器路径换成自己的用于交叉编译的`gcc`路径。
2. 执行`make PORT_DIR=aarch64 XCFLAGS="-O3 -funroll-all-loops -static --param max-inline-insns-auto=550 -DPERFORMANCE_RUN=1" REBUILD=1 run1.log`编译**单核**测试程序；执行`make PORT_DIR=aarch64 XCFLAGS="-O3 -funroll-all-loops -static --param max-inline-insns-auto=550 -DPERFORMANCE_RUN=1 -DMULTITHREAD=4  -DUSE_PTHREAD -pthread" REBUILD=1 run1.log`编译生成**4核**测试程序，其中`-DMULTITHREAD=`参数用于控制生成几核心的测试程序。