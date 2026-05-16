# Adversarial Challenge Templates — FinFlowRL

## Phase 0: Foundation Challenge
When a new decision enters CHP, the adversary MUST address:
1. Why is the proposed direction wrong? (vulnerability_strike)
2. What is the system not seeing? (invalidation_conditions)
3. What is the false consensus risk?

## Domain-Specific Challenges (Finance (Trading))
1. What market regime shifts would invalidate the trading model assumptions?
2. How does this strategy perform under tail risk scenarios (black swan events)?
3. What is the maximum drawdown before position limits are breached?
4. Are there hidden correlations between positions that create concentrated risk?
5. What slippage and latency assumptions are embedded, and what if they're wrong?

## Round 3: Implementation Drift Check
1. Does the implementation match the locked spec acceptance criteria?
2. Are operational handoffs and owner capacity accounted for?
3. Is evidence quality sufficient for the decision domain?

## Council Spawn Triggers
When confidence <85% on high-stakes decisions:
- Attacker Model 1: Challenge foundational assumptions
- Attacker Model 2: Challenge operational feasibility
- Synthesizer: Resolve contradictions and produce final recommendation
