# LangChain 四层架构数据分析智能体（MySQL）— Hybrid Router(缓存+规则+RAG)

这是一个 **可上线、可演进** 的 NL→SQL 数据分析智能体项目：

- **层1：Hybrid Router（缓存 → 规则/词典 → RAG）**
  - L1a 缓存：同用户/角色/归一化问题 TTL 缓存
  - L1b 规则：指标 key/中文名/术语同义词快速命中
  - L1c RAG：Chroma 向量检索召回“指标定义/术语”
- **层2：风险评估（Pre-risk + Post-risk）**
  - Pre-risk：query 文本 + 角色
  - Post-risk：SQL AST（sqlglot）校验 + 关键风险点（敏感列、无 LIMIT 等）
- **层3：受控 Text-to-SQL 兜底**
  - LLM 只输出 **QueryPlan(JSON)**（意图、维度、时间、过滤）
  - 由确定性编译器生成 SQL（保证口径一致与安全边界）
  - validate → repair 循环（最多 N 次）
- **层4：反馈学习**
  - 保存成功查询为 `query_patterns`（可用于后续增强检索/重排）
  - 不自动“生效”为新指标：需要人工审核（避免口径污染）

> 重要：本项目默认优先走“指标口径/模板拼装”，Text-to-SQL 仅做兜底。

## 1) 快速开始（uv）

1. 安装 uv（如果没有）：
```bash
pip install uv
```

2. 创建虚拟环境并安装依赖：
```bash
uv venv
uv sync
```

3. 配置环境变量：
```bash
cp .env.example .env
# 填入 MySQL、LLM、Embedding 信息
```

4. 初始化数据库表：
```bash
uv run python scripts/init_db.py
```

5. 导入示例指标口径与术语（请替换为你自己的事实表/字段）：
```bash
uv run python scripts/seed_assets.py
```

6. 构建向量索引（RAG 用）：
```bash
uv run python scripts/build_vector_index.py
```

7. 启动服务：
```bash
uv run uvicorn app.main:app --reload --port 8000
```

## 2) API

- `GET /health`
- `POST /chat`

请求示例：
```json
{
  "user_id": "u_001",
  "role": "analyst",
  "query": "上周GMV是多少？"
}
```

## 3) 你需要改哪些地方才能适配你的库

- `assets/metrics.seed.json`: 指标口径（fact_table/time_column/measure_expr/default_filters/allowed_dims）
- `assets/terms.seed.json`: 业务术语与同义词
- （进阶）将指标口径、数据字典迁移到 MySQL 中作为治理源，并按版本发布。

