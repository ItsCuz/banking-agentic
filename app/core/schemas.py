from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    message: str = Field(..., min_length=1)


class IntentResult(BaseModel):
    label: str
    confidence: Optional[float] = None
    source: str


class PriorityResult(BaseModel):
    level: str
    reason: str


class PolicyResult(BaseModel):
    intent: str
    snippet: str


class DraftResult(BaseModel):
    reply: str
    missing_information: List[str] = []
    next_action: str
    source: str


class ValidationResult(BaseModel):
    passed: bool
    checks: Dict[str, bool]
    issues: List[str] = []


class RoutingResult(BaseModel):
    decision: str
    reason: str


class WorkflowTrace(BaseModel):
    intent: IntentResult
    priority: PriorityResult
    policy: PolicyResult
    draft: DraftResult
    validation: ValidationResult
    routing: RoutingResult
    engine_mode: str


class AgentResponse(BaseModel):
    final_output: str
    trace: Dict[str, Any]
