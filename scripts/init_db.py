from __future__ import annotations

from app.db import execute


DDL = [
    """
    CREATE TABLE IF NOT EXISTS metric_definitions (
      metric_key        VARCHAR(64) PRIMARY KEY,
      metric_name_zh    VARCHAR(128),
      metric_desc       TEXT,
      fact_table        VARCHAR(128) NOT NULL,
      time_column       VARCHAR(64) NOT NULL,
      measure_expr      TEXT NOT NULL,
      default_filters   TEXT,
      allowed_dims      JSON,
      trigger_keywords  JSON,
      version           INT DEFAULT 1,
      is_active         TINYINT(1) DEFAULT 1,
      updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS column_dictionary (
      table_name        VARCHAR(128) NOT NULL,
      column_name       VARCHAR(128) NOT NULL,
      column_type       VARCHAR(64),
      business_name_zh  VARCHAR(128),
      business_desc     TEXT,
      synonyms          JSON,
      is_sensitive      TINYINT(1) DEFAULT 0,
      PRIMARY KEY (table_name, column_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS query_patterns (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(64),
      user_role VARCHAR(32),
      user_query TEXT,
      metric_key VARCHAR(64),
      sql_text MEDIUMTEXT,
      satisfied TINYINT(1) DEFAULT 0,
      created_at DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]


def main():
    for ddl in DDL:
        execute(ddl)
    # Backfill missing column if table pre-exists
    try:
        execute("ALTER TABLE metric_definitions ADD COLUMN trigger_keywords JSON")
    except Exception:
        pass
    print("OK: tables ensured.")


if __name__ == "__main__":
    main()
