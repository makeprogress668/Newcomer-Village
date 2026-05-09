"""Agent Observability + Evaluation API。

提供"反应式监控"(Observability) + "主动评估"(Evaluation) 接口:
  GET  /api/agents/health    — 7 Agent 健康状态 + 各服务可用性
  POST /api/agents/evaluate  — 对一份现有 TripPlan 做事后评分(支持人工审计)
  GET  /api/agents/metrics   — 关键指标快照 (缓存命中率/调用数等)

也是"人在回路"的接口入口: 用户可以拿到评估报告后决定是否重新生成。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...models.schemas import TripPlan, TripRequest
from ...services.evaluation_service import get_evaluator
from ...services.guardrail_service import get_input_guardrail
from ...services.image_cache import get_cache_stats
from ...services.rag_service import get_rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["AgentOps"])


# ============ Schemas ============

class EvalRequest(BaseModel):
    """对现有 plan 做事后评估的请求。"""
    request: TripRequest
    plan: TripPlan


class GuardrailCheckRequest(BaseModel):
    """单独跑一次输入护栏检查 (用户提交前可预检)。"""
    request: TripRequest


# ============ Endpoints ============

@router.get(
    "/health",
    summary="7 Agent + 关键服务健康状态",
    description="返回各 Agent / 服务的初始化与可用性,供前端展示 / 监控告警",
)
async def agents_health() -> Dict[str, Any]:
    """轻量健康检查。"""
    status = {"overall": "healthy", "components": {}}

    try:
        from ...services.amap_service import get_amap_service
        amap = get_amap_service()
        tools = amap.mcp_tool._available_tools or []
        status["components"]["amap_mcp"] = {
            "ok": len(tools) > 0,
            "tool_count": len(tools),
        }
    except Exception as exc:
        status["components"]["amap_mcp"] = {"ok": False, "error": str(exc)}
        status["overall"] = "degraded"

    try:
        rag = get_rag()
        status["components"]["rag"] = {"ok": True, "doc_count": len(rag.docs)}
    except Exception as exc:
        status["components"]["rag"] = {"ok": False, "error": str(exc)}
        status["overall"] = "degraded"

    try:
        from ...services.amap_rest_service import _api_key
        status["components"]["amap_rest"] = {"ok": bool(_api_key())}
    except Exception:
        status["components"]["amap_rest"] = {"ok": False}

    return status


@router.get(
    "/metrics",
    summary="关键运行指标快照",
    description="图片缓存命中率 / 评估服务可用性等,供 observability 仪表盘消费",
)
async def agents_metrics() -> Dict[str, Any]:
    return {
        "image_cache": get_cache_stats(),
    }


@router.post(
    "/evaluate",
    summary="对一份 TripPlan 做主动质量评估",
    description="返回 5 维度评分 + 总分 + 等级 + 改进警告。"
                "用于 (1) Agent 自动评估 (2) 人工事后审计 (human-in-the-loop)",
)
async def evaluate_plan(req: EvalRequest) -> Dict[str, Any]:
    try:
        report = get_evaluator().evaluate(req.request, req.plan)
        return {"success": True, "data": report.to_dict()}
    except Exception as exc:
        logger.exception("评估失败")
        raise HTTPException(status_code=500, detail=f"评估失败: {exc}")


@router.post(
    "/guardrail/precheck",
    summary="提交规划前做一次输入预检",
    description="人在回路设计: 用户提交前可知道 PII / 敏感词 / 注入风险,改后再提交",
)
async def guardrail_precheck(req: GuardrailCheckRequest) -> Dict[str, Any]:
    result = get_input_guardrail().check(req.request)
    return {
        "success": True,
        "data": {
            "passed": result.passed,
            "severity": result.severity,
            "violations": result.violations,
            "redacted_count": result.redacted_count,
            "sanitized_text": result.sanitized_text,
        },
    }
