# XQUIC Demo 应用程序

本目录包含 XQUIC 的演示客户端和服务器程序，用于展示 XQUIC 库的基本功能和用法。

## 目录内容

- `demo_client.c` - QUIC 客户端演示程序
- `demo_server.c` - QUIC 服务器演示程序
- `common.h` - 公共头文件
- `xqc_hq.*` - HQ (HTTP/0.9) 协议实现

## demo_client.c 功能

**协议支持**: HQ (HTTP/0.9) 和 HTTP/3

**主要功能**:
- 发送 HTTP 请求并接收响应
- 文件下载保存到本地
- 性能统计（传输速度、耗时）
- 多路径 QUIC (Multipath QUIC)
- 0-RTT 早数据支持
- Session Ticket/Token 持久化
- 三种请求模式：
  - SCMR: 单连接多请求
  - SCSR_SERIAL: 多连接串行请求
  - SCSR_CONCURRENT: 多连接并发请求
- iperf 风格带宽报告（每 N 秒输出传输速率）

**命令行参数**:
```
-a <addr>     服务器地址 (默认: 127.0.0.1)
-p <port>     服务器端口 (默认: 8443)
-t <sec>     超时时间 (秒)
-c <algo>    拥塞控制算法 (b:bbr, c:cubic, r:reno)
-C            开启 Pacing
-M            启用多路径 QUIC
-U <url>     请求 URL
-o <dir>     输出文件目录
-0            启用 0-RTT
-l <level>   日志级别 (e:error, d:debug, i:info, w:warn)
-j <sec>     带宽报告间隔秒数 (默认: 1, 0 禁用)
-h            显示帮助信息
```

## demo_server.c 功能

**协议支持**: HQ 和 HTTP/3

**主要功能**:
- 接收请求并返回文件响应
- Dummy 模式（不读文件，返回虚拟数据）
- 多路径 QUIC 支持
- Retry 机制
- 连接统计信息输出
- iperf 风格带宽报告（每 N 秒输出传输速率）

**命令行参数**:
```
-p <port>    监听端口 (默认: 8443)
-c <algo>   拥塞控制 (b:bbr, c:cubic, r:reno)
-C           开启 Pacing
-l <level>  日志级别 (e/d/i/w)
-L <dir>    日志目录
-k <file>   密钥输出文件
-r           启用 retry
-d           Dummy 模式
-M           启用多路径
-6           IPv6
-j <sec>    带宽报告间隔秒数 (默认: 1, 0 禁用)
-h           显示帮助信息
```

## 快速开始

### 1. 编译

确保已按照项目根目录的构建说明编译 XQUIC。

### 2. 生成证书

```bash
cd build/demo
openssl req -newkey rsa:2048 -x509 -nodes -keyout server.key -new -out server.crt \
    -subj /CN=127.0.0.1 -addext "subjectAltName=IP:127.0.0.1" 2>/dev/null
```

### 3. 启动服务端

```bash
./demo_server -l d -p 4433 &
```

### 4. 启动客户端

```bash
./demo_client -a 127.0.0.1 -p 4433 -l d -U "https://127.0.0.1:4433/test" -t 5
```

### 5. 关闭服务端

```bash
kill <server_pid>
```

## 使用示例

### 基本 HTTP 请求

```bash
# 服务端
./demo_server -l d -p 4433

# 客户端
./demo_client -a 127.0.0.1 -p 4433 -U "https://127.0.0.1:4433/testfile" -o ./downloads
```

### 多路径 QUIC

```bash
# 服务端 (启用多路径)
./demo_server -l d -p 4433 -M

# 客户端 (启用多路径)
./demo_client -a 127.0.0.1 -p 4433 -M -U "https://127.0.0.1:4433/test"
```

### 0-RTT 早数据

```bash
# 客户端 (启用 0-RTT)
./demo_client -a 127.0.0.1 -p 4433 -0 -U "https://127.0.0.1:4433/test"
```

## 运行脚本

项目提供了自动化运行脚本：

```bash
# 使用默认配置运行演示
cd xquic/scripts/local_test/demo
./run_demo.sh
```

## 注意事项

1. 客户端和服务端需要在同一台机器或网络可达的环境中运行
2. 默认使用 127.0.0.1 地址，如需远程测试请修改地址
3. 多路径功能需要操作系统支持多个网络接口或绑定不同端口
4. 生产环境请使用正式签名的证书
