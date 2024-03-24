[English](./README.md) | 简体中文

# 环境初始化

```
$ yarn
```

该命令将安装所需要的插件

# 开发

```
$ yarn start
```

此命令启动本地开发服务器并打开浏览器窗口。大多数更改都是实时反映的，无需重新启动服务器。

# 编译

```
$ yarn build
```

此命令将静态内容生成到“build”目录中，并可用于任何静态内容托管服务。编译完成后不支持直接打开.html文件查看，需使用`npm run serve`启动服务。

# 错误处理
1. 若使用`yarn`初始化环境失败，请尝试将yarn升级至最新版本
```
npm install yarn@latest -g
```
2. 若使用`yarn start`启动本地开发服务器失败，请尝试升级nodejs版本
```
#安装n模块
sudo npm install -g n

#升级nodejs到指定版本
sudo n node版本号

#查看nodejs版本(需要重启一下终端)
node -v
```