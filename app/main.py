from fastapi import FastAPI
from app.core.schemas import CustomerRequest, AgentResponse
from app.agent.orchestrator import Orchestrator

app = FastAPI(title="Banking AI Agent")
orchestrator = Orchestrator()

@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: CustomerRequest):
    decision, trace = orchestrator.run_workflow(request.message)
    final_text = f"[{decision}] {trace['draft']}"
    return AgentResponse(final_output=final_text, trace=trace)