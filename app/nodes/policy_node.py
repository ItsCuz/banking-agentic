from app.core.schemas import IntentResult, PolicyResult
from app.data.policies import BANKING_POLICIES


class PolicyNode:
    def process(self, intent: IntentResult) -> PolicyResult:
        snippet = BANKING_POLICIES.get(intent.label, BANKING_POLICIES["general_inquiry"])
        return PolicyResult(intent=intent.label, snippet=snippet)
