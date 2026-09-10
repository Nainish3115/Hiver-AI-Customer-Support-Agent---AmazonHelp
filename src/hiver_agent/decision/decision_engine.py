"""
Main Decision Engine for Phase 7 Auto-Handle vs Escalation Layer.
"""
from typing import Optional, Dict, Any
from hiver_agent.config import DecisionConfig, load_config
from hiver_agent.generation.interface import GenerationResult
from hiver_agent.decision.interface import DecisionInput, DecisionResult
from hiver_agent.decision.policy import HardRulesPolicy, ConfidenceOnlyPolicy, RiskAwarePolicy


class RiskAwareDecisionEngine:
    """
    Main orchestrator for risk-aware decision making.
    Converts generation results into structured decision inputs, evaluates decision policies,
    and returns explainable DecisionResults.
    """
    def __init__(self, config: Optional[DecisionConfig] = None):
        if config is None:
            try:
                app_cfg = load_config()
                config = app_cfg.decision
            except Exception:
                config = DecisionConfig()

        self.config = config
        self.hard_rules_policy = HardRulesPolicy()
        self.risk_aware_policy = RiskAwarePolicy(
            risk_threshold=self.config.risk_threshold,
            min_retrieval_score=self.config.min_retrieval_score,
            high_risk_intents=self.config.high_risk_intents,
            weights=self.config.weights
        )

    def decide(self, inp: DecisionInput, policy_type: str = "risk_aware") -> DecisionResult:
        """
        Execute decision evaluation on a DecisionInput payload using the specified policy_type.
        policy_type choices: 'risk_aware', 'confidence_only', 'hard_rules_only'
        """
        if policy_type == "confidence_only":
            policy = ConfidenceOnlyPolicy(confidence_threshold=self.config.confidence_threshold)
            return policy.evaluate(inp)
        elif policy_type == "hard_rules_only":
            return self.hard_rules_policy.evaluate(inp)
        else:
            return self.risk_aware_policy.evaluate(inp)

    def decide_from_generation_result(
        self,
        gen_res: GenerationResult,
        policy_type: str = "risk_aware"
    ) -> DecisionResult:
        """
        Helper method to evaluate a GenerationResult directly.
        """
        inp = DecisionInput.from_generation_result(gen_res)
        return self.decide(inp, policy_type=policy_type)
