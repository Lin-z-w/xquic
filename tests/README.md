# XQUIC 测试程序

本目录包含 XQUIC 的测试客户端和服务器程序，主要用于功能测试和性能测试。

## 目录内容

- `test_client.c` - QUIC 测试客户端
- `test_server.c` - QUIC 测试服务器
- `platform.h` - 平台相关定义
- `unittest/` - 单元测试

## test_client.c 功能

**协议支持**: HTTP/3、HQ (HTTP/0.9) 和 QUIC Transport

**主要功能**:
- HTTP 请求发送和响应接收
- 性能基准测试
- 多种测试用例 (test_case) 用于测试不同场景
- 多路径 QUIC (Multipath QUIC) 支持
- Datagram (不可靠数据传输) 支持
- H3 Extended Datagram 支持
- 流量控制算法：BBR, Cubic, Reno, COPA
- 多连接并发测试
- 数据校验 (echo check)
- 0-RTT 早数据支持
- 丢包模拟测试

**命令行参数**:
```
-a <addr>      服务器地址 (默认: 127.0.0.1)
-p <port>     服务器端口 (默认: 8443)
-t <sec>      超时时间 (秒)
-c <algo>     拥塞控制 (b:bbr, c:cubic, r:reno, P:copa)
-C             开启 Pacing
-n <num>      连接数 (默认: 100)
-k <num>      并发请求数 (默认: 1)
-s <size>     发送 body 大小 (字节)
-M             启用多路径 QUIC
-U <num>      发送 datagram (0:关闭, 1:单发, 2:批量)
-x <case>     测试用例 ID
-d <rate>     丢包率 (0-1000)
-e             启用 echo 检查
-0             启用 0-RTT
-l <level>    日志级别 (e:error, d:debug)
-h             显示帮助信息
```

### 测试用例 (test_case)

客户端支持多种测试用例，通过 `-x` 参数指定：

- `3` - Server Initial DCID 损坏
- `4` - Server Initial SCID 损坏
- `5` - Socket 发送失败
- `9` - 重复包
- `12` - Linger close
- `14` - 请求关闭
- `15` - 空闲后重启
- `16` - 应用延迟
- `17` - 服务器发起流
- `21` - Reset stream
- `22` - Client Initial DCID 损坏
- `23` - Client Initial SCID 损坏
- `33` - Version negotiation
- `39` - 0-RTT buffer 测试
- `41` - Handshake 后 Stateless reset
- `45` - Handshake 中 Stateless reset
- `46` - Initial 丢包测试 0-RTT

## test_server.c 功能

**协议支持**: HTTP/3、HQ 和 QUIC Transport

**主要功能**:
- 接收请求并返回文件响应
- Echo 模式 (回显接收的数据)
- Dummy 模式 (返回虚拟数据)
- 多路径 QUIC 支持
- Datagram 支持
- H3 Extended Datagram 支持
- Load Balancer CID 加密
- 连接统计信息 (send_count, lost_count, srtt 等)

**命令行参数**:
```
-a <addr>     监听地址 (默认: 127.0.0.1)
-p <port>     监听端口 (默认: 8443)
-c <algo>     拥塞控制 (b:bbr, c:cubic, r:reno, B:bbr2)
-C             开启 Pacing
-s <size>     发送 body 大小 (字节)
-e             Echo 模式 (回显)
-w <file>     保存接收 body 到文件
-r <file>     从文件读取发送 body
-l <level>    日志级别 (e/d)
-x <case>     测试用例 ID
-M             启用多路径
-R <num>      启用 reinjection (1/2/4)
-U <num>      发送 datagram (0/1/2)
-L             无限发送模式
-m             mpshell 模式
-h             显示帮助信息
```

## 快速开始

### 1. 编译

确保已按照项目根目录的构建说明编译 XQUIC。

### 2. 生成证书

```bash
cd build/tests
openssl req -newkey rsa:2048 -x509 -nodes -keyout server.key -new -out server.crt \
    -subj /CN=127.0.0.1 -addext "subjectAltName=IP:127.0.0.1" 2>/dev/null
```

### 3. 基本测试

```bash
# 启动服务端
./test_server -l d -p 8443 -c b &

# 启动客户端 (单请求)
./test_client -a 127.0.0.1 -p 8443 -c b -n 1 -k 1 -s 1024000

# 关闭服务端
kill %1
```

### 4. 性能测试

```bash
# 服务端 (BBR, 1MB 响应)
./test_server -l d -p 8443 -c b -s 1048576 &

# 客户端 (100 连接, 10 并发, 10MB 请求)
./test_client -a 127.0.0.1 -p 8443 -c b -n 100 -k 10 -s 10485760
```

### 5. 多路径测试

```bash
# 服务端 (启用多路径)
./test_server -l d -p 8443 -M &

# 客户端 (启用多路径)
./test_client -a 127.0.0.1 -p 8443 -M -n 1 -k 1
```

### 6. Datagram 测试

```bash
# 服务端 (发送 datagram)
./test_server -l d -p 8443 -U 1 &

# 客户端 (发送 datagram)
./test_client -a 127.0.0.1 -p 8443 -U 1
```

### 7. Echo 测试

```bash
# 服务端 (Echo 模式)
./test_server -l d -p 8443 -e &

# 客户端 (Echo 检查)
./test_client -a 127.0.0.1 -p 8443 -e -s 1024000
```

## 使用示例

### 测试特定功能

```bash
# 测试 0-RTT
./test_server -l d -p 8443 &
./test_client -a 127.0.0.1 -p 8443 -0 -x 39

# 测试丢包恢复
./test_server -l d -p 8443 &
./test_client -a 127.0.0.1 -p 8443 -d 50

# 测试多连接
./test_server -l d -p 8443 &
./test_client -a 127.0.0.1 -p 8443 -n 50 -k 10
```

### 集成测试脚本

项目提供了自动化测试脚本：

```bash
# 运行单元测试
cd build
make run_tests
./tests/run_tests

# 运行集成测试 (需要先启动服务端)
./tests/test_server -l d -p 8443 &
sleep 1
./tests/test_client -a 127.0.0.1 -p 8443
```

## 统计信息解读

客户端输出：
```
>>>>>>>> request time cost:12345 us, speed: 8192 Kbit/s 
>>>>>>>> send_body_size:1024000, recv_body_size:1024000 
retx:0, sent:10, max_pto:0
```

- `request time cost`: 请求耗时 (微秒)
- `speed`: 传输速度 (Kbit/s)
- `send_body_size`: 发送数据量
- `recv_body_size`: 接收数据量
- `retx`: 重传次数
- `sent`: 发送数据包数
- `max_pto`: 最大PTO退避时间

服务端输出：
```
send_count:100, lost_count:0, tlp_count:0, recv_count:50, srtt:1000
early_data_flag:1, conn_err:0
```

- `send_count`: 发送数据包数
- `lost_count`: 丢包数
- `tlp_count`: Tail Loss Probe 次数
- `recv_count`: 接收数据包数
- `srtt`: 平滑往返时间 (毫秒)
- `early_data_flag`: 0-RTT 早数据标志

## 注意事项

1. 测试客户端和服务端需要配合使用
2. 某些测试用例需要特定网络条件
3. 丢包测试 (`-d` 参数) 可能导致测试失败，属于正常现象
4. 多路径测试需要多个网络接口或端口
5. 生产环境请使用正式签名的证书
