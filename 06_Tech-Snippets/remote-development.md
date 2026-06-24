# Remote Development Checklist

## SSH Config
在本机 `~/.ssh/config` 中维护别名，避免到处复制 IP 和密钥路径：

```sshconfig
Host <HOST_ALIAS>
  HostName <SERVER_IP>
  User <USER_NAME>
  IdentityFile <PRIVATE_KEY_PATH>
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

连接：

```bash
ssh <HOST_ALIAS>
```

## Workspace Layout
```text
~/workspace/
  project-a/
  project-b/
  logs/
  datasets/
```

建议把代码、数据、日志分开，避免清理实验输出时误删源码。

## Long Running Jobs
```bash
tmux new -s <SESSION_NAME>
tmux ls
tmux attach -t <SESSION_NAME>
```

## File Transfer
小文件可以用 `scp`：

```bash
scp <LOCAL_FILE> <HOST_ALIAS>:<REMOTE_PATH>
scp <HOST_ALIAS>:<REMOTE_FILE> <LOCAL_PATH>
```

大目录同步优先记录源路径和目标路径，执行前先 dry run：

```bash
rsync -av --dry-run <LOCAL_DIR>/ <HOST_ALIAS>:<REMOTE_DIR>/
```

## Safety Notes
- 删除或移动大目录前先 `pwd`、`ls`、`du -sh` 确认路径。
- 私钥权限保持最小化，不上传到共享服务器。
- 端口转发只绑定需要的服务，避免暴露调试端口。
