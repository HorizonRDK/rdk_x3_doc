---
sidebar_position: 2
---

# scp命令

Linux scp 命令用于 Linux 之间复制文件和目录。

scp 是 secure copy 的缩写, scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。

scp 是加密的，[rcp](https://www.runoob.com/linux/linux-comm-rcp.html) 是不加密的，scp 是 rcp 的加强版。

## 语法

```
scp [-346BCpqrTv] [-c cipher] [-F ssh_config] [-i identity_file]
            [-J destination] [-l limit] [-o ssh_option] [-P port]
            [-S program] source ... target
```

简易写法:

```
scp [可选参数] file_source file_target 
```

## 参数说明

- -3：通过本地主机传输两个远程主机之间的文件。如果不使用此选项，数据将直接在两个远程主机之间传输。请注意，此选项会禁用传输进度显示。
- -4： 强制scp命令只使用IPv4寻址
- -6： 强制scp命令只使用IPv6寻址
- -B： 选择批处理模式（防止询问密码或密码短语）
- -C： 允许压缩。（将-C标志传递给ssh，从而打开压缩功能）
- -p：保留原文件的修改时间，访问时间和访问权限。
- -q： 静默模式：禁用进度表以及来自ssh(1)的警告和诊断消息。
- -r： 递归地复制整个目录。请注意，scp会遵循树遍历中遇到的符号链接。、
- -T：禁用严格的文件名检查。默认情况下，当从远程主机复制文件到本地目录时，scp会检查接收的文件名是否与命令行上请求的文件名匹配，以防止远程端发送意外或不需要的文件。由于不同操作系统和Shell解释文件名通配符的方式不同，这些检查可能会导致希望的文件被拒绝。此选项禁用这些检查，但需要完全信任服务器不会发送意外的文件名。
- -v：详细模式。导致scp和ssh(1)打印关于它们的进展的调试消息。这在调试连接、身份验证和配置问题时很有帮助。
- -c cipher： 选择用于加密数据传输的密码。此选项直接传递给ssh(1)。
- -F ssh_config： 指定用于ssh的替代每用户配置文件。此选项直接传递给ssh(1)。
- -i identity_file： 选择用于公钥身份验证的身份（私钥）文件。此选项直接传递给ssh(1)。
- -l limit： 限制使用的带宽，以Kbit/s为单位。
- -o ssh_option： 可以用于以ssh_config(5)中使用的格式传递选项给ssh。这对于指定没有单独的scp命令行标志的选项非常有用。
- -P port：指定要连接到远程主机的端口。请注意，此选项使用大写的 'P'，因为小写的 '-p' 已经被保留用于保留文件的修改时间和模式。
- -S program： 用于加密连接的程序名称。该程序必须理解ssh(1)选项。
