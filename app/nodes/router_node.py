from app.core.schemas import DraftResult, PriorityResult, RoutingResult, ValidationResult


class RouterNode:
    def process(
        self,
        priority: PriorityResult,
        validation: ValidationResult,
        draft: DraftResult,
    ) -> RoutingResult:
        if priority.level == "High":
            return RoutingResult(decision="escalate_to_human", reason="High-priority banking risk.")
        if not validation.passed:
            return RoutingResult(decision="ask_for_more_information", reason="Draft did not pass validation.")
        if draft.missing_information:
            return RoutingResult(decision="ask_for_more_information", reason="Required customer details are missing.")
        return RoutingResult(decision="send_reply", reason="Draft is grounded, complete, and low enough risk.")
