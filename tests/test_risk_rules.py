from app.layer2_risk import assess_post_risk
from app.config import settings


def test_risk_blocks_multiple_statements():
    settings.risk_block_multiple_statements = True
    r = assess_post_risk("SELECT 1; SELECT 2")
    assert r.action == "block"


def test_risk_cross_join_requires_approval():
    settings.risk_block_cross_join = True
    r = assess_post_risk("SELECT * FROM a CROSS JOIN b LIMIT 10")
    assert r.action == "require_approval"


def test_risk_union_limit():
    settings.risk_max_unions = 0
    r = assess_post_risk("SELECT 1 UNION SELECT 2 LIMIT 10")
    assert r.action == "require_approval"
