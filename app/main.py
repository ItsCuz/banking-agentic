from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.agent.orchestrator import Orchestrator
from app.core.schemas import AgentResponse, CustomerRequest
from app.core.settings import settings

app = FastAPI(title="Banking AI Agent", version="1.0.0")
orchestrator = Orchestrator()


@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: CustomerRequest):
    return await run_agent_endpoint(request)


@app.post("/run-agent", response_model=AgentResponse)
async def run_agent_endpoint(request: CustomerRequest):
    routing, trace = orchestrator.run_workflow(request.message)
    prefix = {
        "send_reply": "SEND",
        "ask_for_more_information": "ASK_MORE",
        "escalate_to_human": "ESCALATE",
    }.get(routing.decision, routing.decision.upper())
    return AgentResponse(final_output=f"[{prefix}] {trace.draft.reply}", trace=trace.model_dump())


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "banking-ai-agent"}


@app.get("/config")
async def config():
    return {
        "ollama_url": settings.OLLAMA_URL,
        "ollama_model": settings.MODEL_NAME,
        "intent_service_host": settings.INTENT_SERVICE_HOST,
        "intent_service_port": settings.INTENT_SERVICE_PORT,
        "use_grpc_intent": settings.USE_GRPC_INTENT,
        "intent_model_config": settings.INTENT_MODEL_CONFIG,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(
        """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Banking AI Agent</title>
  <style>
    :root { color-scheme: light; --ink:#172033; --muted:#5f6b7a; --line:#d8dee8; --brand:#0f766e; --risk:#b42318; --bg:#f7f9fb; }
    body { margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:22px 28px; background:#ffffff; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:24px; letter-spacing:0; }
    main { max-width:1180px; margin:24px auto; padding:0 18px; display:grid; grid-template-columns: minmax(280px, 420px) 1fr; gap:18px; }
    section, .node { background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px; }
    textarea { width:100%; min-height:160px; box-sizing:border-box; border:1px solid var(--line); border-radius:8px; padding:12px; font-size:15px; resize:vertical; }
    button { margin-top:12px; border:0; border-radius:8px; padding:11px 16px; background:var(--brand); color:#fff; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.6; cursor:wait; }
    .samples button { width:100%; margin-top:8px; text-align:left; background:#eef8f6; color:#115e59; border:1px solid #b7ddd7; }
    .trace { display:grid; gap:10px; }
    .node h3 { margin:0 0 8px; font-size:15px; }
    .node p { margin:0; color:var(--muted); line-height:1.45; white-space:pre-wrap; }
    .badge { display:inline-block; padding:3px 8px; border-radius:999px; background:#e7f5ef; color:#067647; font-size:12px; font-weight:700; }
    .risk { background:#fee4e2; color:var(--risk); }
    .output { margin-top:12px; padding:12px; background:#f2f4f7; border-radius:8px; line-height:1.5; }
    @media (max-width: 860px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header><h1>Banking AI Agentic Workflow</h1></header>
  <main>
    <section>
      <h2>Customer message</h2>
      <textarea id="message">I transferred 2,000,000 VND this morning. My account was debited but the receiver has not received the money. Transaction ID is TX123456.</textarea>
      <button id="run" onclick="runWorkflow()">Run workflow</button>
      <div class="samples">
        <button onclick="fillSample('Someone called me asking for my OTP and said they are from the bank. I suspect fraud and need urgent help.')">Fraud alert</button>
        <button onclick="fillSample('My account is blocked after I entered the wrong password three times. Please help me unlock it.')">Blocked account</button>
        <button onclick="fillSample('I applied for a new ATM card two weeks ago but have not received it yet.')">Card not received</button>
      </div>
      <div id="final" class="output">Final output will appear here.</div>
    </section>
    <section>
      <h2>Node trace</h2>
      <div id="trace" class="trace"></div>
    </section>
  </main>
  <script>
    function fillSample(text) { document.getElementById('message').value = text; }
    function node(title, content, badge, risk=false) {
      return `<div class="node"><h3>${title} ${badge ? `<span class="badge ${risk ? 'risk' : ''}">${badge}</span>` : ''}</h3><p>${content}</p></div>`;
    }
    async function runWorkflow() {
      const button = document.getElementById('run');
      const message = document.getElementById('message').value.trim();
      if (!message) return;
      button.disabled = true;
      document.getElementById('trace').innerHTML = '';
      document.getElementById('final').innerText = 'Processing...';
      try {
        const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message}) });
        const data = await res.json();
        const t = data.trace;
        document.getElementById('final').innerText = data.final_output;
        document.getElementById('trace').innerHTML =
          node('1. Intent Detection', `label: ${t.intent.label}\\nsource: ${t.intent.source}`, t.intent.label) +
          node('2. Priority Detection', `${t.priority.reason}`, t.priority.level, t.priority.level === 'High') +
          node('3. Policy Retrieval', t.policy.snippet, t.policy.intent) +
          node('4. Response Drafting', `${t.draft.reply}\\n\\nmissing_information: ${t.draft.missing_information.join(', ') || 'none'}\\nsource: ${t.draft.source}`, t.draft.next_action) +
          node('5. Validation', `passed: ${t.validation.passed}\\nissues: ${t.validation.issues.join(', ') || 'none'}`, t.validation.passed ? 'PASS' : 'FAIL', !t.validation.passed) +
          node('6. Escalation Routing', t.routing.reason, t.routing.decision, t.routing.decision === 'escalate_to_human');
      } catch (e) {
        document.getElementById('final').innerText = 'Request failed: ' + e.message;
      } finally {
        button.disabled = false;
      }
    }
  </script>
</body>
</html>
        """
    )
