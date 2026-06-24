# Linux Server Operations

## Scope
服务器日常巡检、服务排障和进程管理的常用命令。命令中的真实路径、端口和服务名使用占位符。

## System Snapshot
```bash
hostname
date
uptime
who
uname -a
```

## Disk And Memory
```bash
df -h
du -sh <PATH>
free -h
top
```

## Processes
```bash
ps aux | grep <KEYWORD>
pgrep -af <KEYWORD>
```

结束进程前先确认 PID 和命令行：

```bash
kill <PID>
```

如果普通 `kill` 无效，再判断是否需要更强制的处理。

## Network
```bash
ss -ltnp
ss -ltnp | grep ":<PORT>"
curl -v http://127.0.0.1:<PORT>/health
```

## Logs
```bash
journalctl -u <SERVICE_NAME> -n 100 --no-pager
journalctl -u <SERVICE_NAME> -f
tail -n 200 <LOG_FILE>
```

## Systemd Service
```bash
sudo systemctl status <SERVICE_NAME>
sudo systemctl restart <SERVICE_NAME>
sudo systemctl enable <SERVICE_NAME>
```

## Deployment Notes
- 记录服务目录、启动命令、配置文件路径和日志路径。
- 变更前保存当前版本、commit、配置摘要和回滚方式。
- 不把 token、密码或私钥写进 systemd unit；优先使用环境文件，并确保权限最小化。
