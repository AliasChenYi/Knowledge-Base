GitHub SSH 配置指南
配置 SSH 后，你将不再需要频繁输入 GitHub 的账号密码，且数据传输更安全、稳定。

1. 检查本地是否有现有的 SSH 密钥
在终端输入以下命令查看是否存在 id_rsa.pub 或 id_ed25519.pub：

ls -al ~/.ssh
如果看到相关文件，你可以直接跳过生成步骤。

2. 生成新的 SSH 密钥
如果目录下没有密钥文件，请执行以下命令（建议使用 ed25519 算法，安全性更高）：

ssh-keygen -t ed25519 -C "your_email@example.com"
提示：系统会询问存放路径，直接按 Enter 使用默认值。

提示：系统会询问 Passphrase（密码短语），按 Enter 留空即可（如果不留空，每次推送都需要输入该密码）。

3. 将 SSH 公钥添加到 GitHub
第一步：复制公钥内容

cat ~/.ssh/id_ed25519.pub
将终端输出的以 ssh-ed25519 开头的字符串全部复制下来。

第二步：在 GitHub 页面配置
登录 GitHub，点击右上角头像 -> Settings。

在左侧侧边栏中，点击 SSH and GPG keys。

点击右上角的 New SSH key 按钮。

Title 随便填（例如 Dinglab-V100），将刚才复制的公钥粘贴到 Key 文本框中。

点击 Add SSH key。

4. 验证连接
在终端输入以下命令：

ssh -T git@github.com
如果你看到如下提示，说明配置成功：

"Hi [YourUsername]! You've successfully authenticated, but GitHub does not provide shell access."

5. 将仓库地址切换为 SSH 模式
由于你现在的仓库是用 HTTPS 克隆的，执行以下命令切换到 SSH 协议：

# 切换 remote origin 地址
git remote set-url origin git@github.com:AliasChenYi/Knowledge-Base.git