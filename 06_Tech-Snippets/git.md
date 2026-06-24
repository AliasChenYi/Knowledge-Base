# GitHub SSH 配置指南

## Scope
配置 SSH 后，推送和拉取 GitHub 仓库时不需要反复输入账号密码，传输也更稳定。

## 1. 检查现有密钥
查看是否已经存在 `id_rsa.pub` 或 `id_ed25519.pub`：

```bash
ls -al ~/.ssh
```

如果看到可用公钥，可以直接跳过生成步骤。

## 2. 生成新密钥
如果目录下没有密钥文件，建议使用 `ed25519`：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- 存放路径：直接按 Enter 使用默认值。
- Passphrase：个人机器可以留空；共享机器建议设置。

## 3. 添加公钥到 GitHub
复制公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

将终端输出的以 `ssh-ed25519` 开头的字符串全部复制下来。

在 GitHub 页面进入：

`Settings` -> `SSH and GPG keys` -> `New SSH key`

`Title` 可填写机器名，例如 `Dinglab-V100`；`Key` 粘贴刚才复制的公钥。

## 4. 验证连接

```bash
ssh -T git@github.com
```

如果你看到如下提示，说明配置成功：

```text
Hi [YourUsername]! You've successfully authenticated, but GitHub does not provide shell access.
```

## 5. 将仓库地址切换为 SSH
如果仓库当前使用 HTTPS，可以切换到 SSH：

```bash
git remote set-url origin git@github.com:AliasChenYi/Knowledge-Base.git
git remote -v
```

## Troubleshooting
- `Permission denied (publickey)`：确认 GitHub 中的公钥和本机私钥匹配。
- `Host key verification failed`：确认 `~/.ssh/known_hosts` 中的 GitHub 记录是否异常。
- 多个 GitHub 账号：在 `~/.ssh/config` 中为不同账号配置不同 `Host` alias。
