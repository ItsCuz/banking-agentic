from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CustomerRequest(BaseModel):
    message: str

class NodeOutput(BaseModel):
    intent: Optional[str] = None
    priority: Optional[str] = None
    policy: Optional[str] = None
    draft_reply: Optional[str] = None
    validation_status: Optional[bool] = None
    decision: Optional[str] = None

class AgentResponse(BaseModel):
    final_output: str
    trace: Dict[str, Any]