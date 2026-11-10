# 本地运行数据

Bookworm 蠹将运行数据与源码分离；除本说明外，本目录不提交 Git。

- `cache/http.sqlite`：WoS / OpenAlex 响应缓存。
- `runs/<run>/`：语料、检查点、请求和结果 JSON、审计日志、参考文献导出及可选 Obsidian 笔记。
- `archive/`：保留的历史研究材料、PDF 和旧辅助文件，不由运行程序修改。

新环境使用 `BOOKWORM_CACHE_DIR` 和 `BOOKWORM_RUNS_DIR`，兼容旧 `SCIB_CACHE_DIR` / `SCIB_CORPORA_DIR`。
相对目录基于 `--home`、`BOOKWORM_HOME` 或当前工作目录解析。

Bookworm 2.0 不再自动删除历史运行。失败、中断与未完成任务保留检查点和缓存。
新 `run` / `task` 默认保留缓存；低层命令遵从 `BOOKWORM_KEEP_CACHE`（兼容 `SCIB_KEEP_CACHE`）及 `--keep-cache`。
`.bookworm-run.json` 标识运行状态；旧 `.scibooster-run.json` 保留原样。
续接语料始终写入新目录，`checkpoint.json` 可能含尚未筛选的候选，需查看评分后使用。
