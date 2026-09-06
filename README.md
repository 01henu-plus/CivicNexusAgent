# CivicNexusAgent

一个可用于实习面试演示的“城市公共服务事项”多 Agent 项目。用户用中文描述
污水、道路、垃圾等问题，系统会完成受理、案例检索、部门路由、复核和本地
事项创建。实现刻意保持在一个小型 Harness 内，方便逐文件讲解。

## 核心链路

```text
用户消息
  -> AgentRuntime（Harness）
  -> Intake / CaseAnalysis / Routing / Review Agent
  -> PolicyGate 校验 proposal 与工具权限
  -> ToolRegistry 执行工具（检索、地址校验、幂等建单）
  -> StateReducer 更新 TaskContext
  -> EventLog + FactLedger + 分层 Memory
  -> ContextManager 压缩并保存 checkpoint
```

重点可展示：

- 多 Agent：每个 Agent 只读 `CompiledContext` 并返回 `AgentProposal`，不能直接改状态。
- 状态管理：`TaskStatus` 与显式转移表；Harness 统一执行、预算和复核循环。
- 上下文管理：保留结构化 facts、最近消息和摘要，按字符预算压缩；checkpoint 可恢复。
- 工具调用：技能 manifest 声明允许的工具；PolicyGate 拒绝越权调用；建单使用幂等键。
- 记忆管理：task/session/user 三种命名空间，候选画像只有用户明确确认后才写入。
- 动态 Skill：读取 `configs/skills/*.yaml`，按触发词和优先级选择技能，无需改运行时代码。
- 可观测与评测：append-only trace、事件重放、运行指标和固定 Gold 集。

## 目录结构

```text
CivicNexusAgent/
├── configs/skills/            # 动态 Skill manifest
├── city_data/                 # 历史案例样例与检索数据
├── src/civicnexus/
│   ├── agents/                # 受理、分析、路由、复核 Agent
│   ├── api/                   # FastAPI 用户与管理员接口
│   ├── domain/                # TaskContext、状态和 proposal 模型
│   ├── evaluation/            # Gold 集、评测 runner、运行指标
│   ├── memory/                # 事件、事实、分层记忆协议与内存实现
│   ├── persistence/           # SQLAlchemy repository、快照与缓存适配器
│   ├── retrieval/             # BM25 / BGE-M3 混合检索
│   ├── runtime/               # Harness、状态机、上下文压缩
│   ├── skills/                # Skill 注册和选择
│   └── tools/                 # 工具注册、权限和内置工具
├── frontend/                  # Vue 3 + Vite：ChatGPT 风格用户端与独立管理端
└── tests/                     # 单元与集成测试
```

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m uvicorn civicnexus.main:app --reload
```

另开终端启动前端：

```powershell
cd frontend
npm install
npm run dev
```

- API：<http://127.0.0.1:8000>
- Swagger：<http://127.0.0.1:8000/docs>
- 前端：<http://127.0.0.1:5173>
- 首次启动会预置一个管理员账号：`admin` / `change-me`（可通过 `ADMIN_USERNAME`、`ADMIN_PASSWORD` 修改）。
- 默认普通用户账号：`demo` / `demo-me`（可通过 `USER_USERNAME`、`USER_PASSWORD`、`USER_DISPLAY_NAME` 修改）。

用户端只呈现对话、回复、输入和加载状态，布局采用 ChatGPT 式深色会话栏与中央消息流；
访问 `/login` 使用统一表单登录，也可以选择以访客继续。系统根据账号权限自动进入
对话页或运行监控页；登录页提供“注册账户”入口，注册账号始终是普通用户，
不会看到运行 Trace、事实账本或记忆字段。

默认使用本地 SQLite、内存缓存和规则/模拟 LLM，离线即可演示主链路。生产或
Docker 配置可切换到 PostgreSQL、Redis 和 OpenAI-compatible LLM。

## API 与评测

用户接口：

- `POST /api/v1/auth/login` 统一登录（返回账号角色）
- `POST /api/v1/auth/register` 注册普通用户并自动登录
- `GET /api/v1/auth/me` 获取当前登录身份
- `POST /api/v1/tasks` 创建事项
- `POST /api/v1/tasks/{task_id}/messages` 继续对话
- `GET /api/v1/tasks/{task_id}/messages` 获取用户可见消息

- `GET /api/v1/admin/overview`：任务/事件/建单概览
- `GET /api/v1/admin/tasks/{task_id}`：trace、状态、上下文、记忆、技能和快照
- `GET /api/v1/admin/metrics`：从数据库事件和上下文计算的运行指标
- `GET /api/v1/admin/evaluation/report`：运行固定 `civicnexus-gold-v1` Gold 集

Gold 评测包含 6 条固定中文案例，输出分类准确率、地点准确率、Skill/路由准确率、
Macro-F1 和完全匹配率。运行指标只根据实际事件、快照和任务数据计算；空数据库
返回 `0`（而不是伪造满分），并在响应中提供 evidence 计数，便于面试时解释指标来源。

## 测试与质量检查

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q
python -m ruff check src tests
```

当前测试覆盖状态机、上下文压缩/恢复、事实版本、画像确认、记忆隔离、幂等建单、
Gold 评测和运行指标。

## 外部检索服务（可选）

项目支持 OpenAI-compatible `/embeddings` 的 BGE-M3 服务。设置 `.env` 中的
`EMBEDDING_BASE_URL` 和密钥后，可构建历史案例索引：

```powershell
python -m civicnexus.retrieval.indexer
```

未配置时仍可使用本地 BM25/规则回退完成演示。
本地基础安装不强制编译 ChromaDB；需要在线向量检索时安装可选依赖：

```powershell
python -m pip install -e ".[retrieval]"
```

Docker 环境可以在启动 API 前预先构建检索索引，不需要在宿主机安装 Python
或 ChromaDB：

```powershell
docker compose up -d --build
docker compose run --rm --no-deps api python -m civicnexus.retrieval.indexer `
  --input /app/city_data/cases_normalized.jsonl
```

该命令只处理 `retrieval` 划分中的案例，将 BM25 文件写入
`civicnexus_retrieval_data` 卷，并通过 `civicnexus-chroma` 服务将 BGE-M3 向量写入
`civicnexus_chroma_data` 卷。PostgreSQL 只保存任务、事件、事实、快照和本地演示事项，
不保存向量。
如果案例文本发生变化并且需要重新计算已有案例的向量，增加
`--reset-vector-index` 参数。

## Docker

```powershell
docker compose up --build
```

Compose 会启动 API、Vue 前端、PostgreSQL 和 Redis；密钥只放在本地 `.env`，不要提交。

后端主体源码控制在约 5000 行以内（硬上限 5500 行），前端代码不计入该限制。
