from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = ""
    mysql_password: str = ""
    mysql_database: str = ""

    # LLM (OpenAI-compatible)
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    # Embeddings (OpenAI-compatible)
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_model: str = "text-embedding-3-small"

    # App
    app_env: str = "dev"
    app_port: int = 8000
    chroma_dir: str = ".chroma"
    log_level: str = "INFO"

    # Router thresholds
    router_conf_rule: float = 0.90
    router_conf_rag: float = 0.80
    router_conf_ask: float = 0.60

    # Safety limits
    max_rows: int = 10_000
    max_repair_rounds: int = 2

    # Risk rules (comma-separated lists)
    risk_block_keywords: str = "drop,truncate,delete,update,insert,alter,create"
    risk_block_keywords_zh: str = "删除,清空,改表,建表,更新"
    risk_sensitive_columns: str = "salary,phone,id_card,email,address"
    risk_restricted_keywords: str = "财务,薪资,salary,revenue"
    risk_restricted_roles: str = "intern,guest"
    risk_require_limit: bool = True
    risk_max_query_length: int = 8000
    risk_max_tables: int = 8
    risk_max_joins: int = 6
    risk_max_subqueries: int = 4
    risk_max_unions: int = 2
    risk_block_cross_join: bool = True
    risk_block_multiple_statements: bool = True

    # L3 control thresholds
    l3_min_sql_confidence: float = 0.2
    l3_require_confirmation_sql_confidence: float = 0.4


settings = Settings()
