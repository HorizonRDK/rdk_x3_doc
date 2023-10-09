---
sidebar_position: 2
---

# 2.2 系统更新
出于系统安全、稳定性的考虑，推荐用户安装完系统后，通过`apt`命令对系统进行更新。

在`/etc/apt/source.list`文件中，保存了`apt`命令的软件源列表，在安装软件前，需要先通过`apt`命令更新package列表。

首先打开终端命令行，输入如下命令：
```bash
sudo apt update
```
其次，升级所有已安装的软件包到最新版本，命令如下：
```bash
sudo apt full-upgrade
```

:::tip
推荐使用`full-upgrade`而不是`upgrade`选项，这样当相关依赖发生变动时，也会同步更新依赖包。

当运行`sudo apt full-upgrade`命令时，系统会提示数据下载和磁盘占用大小，但是`apt`不会检查磁盘空间是否充足，建议用户通过`df -h`命令手动检查。此外，升级过程中下载的deb文件会保存在`/var/cache/apt/archives`目录中，用户可以通过`sudo apt clean`命令删除缓存文件以释放磁盘空间。
:::

执行`apt full-upgrade`命令后，可能会重新安装驱动、内核文件和部分系统软件，建议用户手动重启设备使更新生效，命令如下：

```bash
sudo reboot
```