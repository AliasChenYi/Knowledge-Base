# 06 Tech Snippets

## Scope
沉淀高频命令、环境配置、服务部署、排障流程和一次踩坑后的修复步骤。

## Suggested Topics
- Git / SSH / remote development
- Linux service deployment and process management
- Codex / CLI / API proxy configuration
- GPU server environment checks
- Networking, ports, firewall, and reverse proxy notes

## Core Files
- `git.md`：GitHub SSH 配置与排障。
- `remote-development.md`：SSH alias、tmux、文件传输和远程工作区规范。
- `linux-server.md`：Linux 服务器巡检、进程、网络、日志和 systemd。
- `gpu-server-checklist.md`：GPU 环境、进程、显存和常见 CUDA 问题。
- `搭建服务.md`：CLIProxyAPI 服务搭建与 Codex 客户端配置。

## Snippet Standard
- 写清楚适用平台、前置条件和验证命令。
- 命令块保持可复制运行，变量用 `<PLACEHOLDER>` 表示。
- 记录失败症状、排查命令和最终修复方式。
- 不保存 API key、token、订阅链接、私钥或真实密码。

## File Naming
- 使用短横线或中文短标题均可，但同类主题保持一致。
- 部署类笔记建议包含对象名，例如 `codex-cliproxyapi-setup.md`。
