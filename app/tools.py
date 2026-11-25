"""The three tools the claims agent can use, backed by in-memory fake data.

Nothing here touches a network or database, so tool results are always the same.
"""
from __future__ import annotations

from typing import Any, Callable

POLICIES: dict[str, dict[str, Any]] = {
    "POL-100234": {
        "policy_id": "POL-100234",
        "type": "auto",
        "holder_first_name": "Jordan",
        "deductible": 500,
        "coverage_limit": 50000,
        "coverages": ["liability", "collision", "comprehensive"],
        "rental_reimbursement": True,
    },
    "POL-200567": {
        "policy_id": "POL-200567",
        "type": "home",
        "holder_first_name": "Priya",
        "deductible": 1000,
        "coverage_limit": 350000,
        "coverages": ["dwelling", "personal_property", "additional_living_expenses"],
        "rental_reimbursement": False,
    },
    "POL-300891": {
        "policy_id": "POL-300891",
        "type": "auto",
        "holder_first_name": "Sam",
        "deductible": 250,
        "coverage_limit": 50000,
        "coverages": ["liability", "collision"],
        "rental_reimbursement": False,
    },
}

CLAIMS: dict[str, dict[str, Any]] = {
    "CLM-2001": {"claim_id": "CLM-2001", "policy_id": "POL-100234", "status": "under review",
                 "amount": 4200, "adjuster": "A. Patel", "next_step": "upload repair estimate"},
    "CLM-2002": {"claim_id": "CLM-2002", "policy_id": "POL-200567", "status": "approved",
                 "amount": 6000, "adjuster": "L. Chen", "next_step": "payment will be issued within 5 business days"},
    "CLM-2003": {"claim_id": "CLM-2003", "policy_id": "POL-300891", "status": "pending documents",
                 "amount": 1800, "adjuster": "R. Okafor", "next_step": "send photos of the damage"},
}


def lookup_policy(policy_id: str) -> dict[str, Any]:
    """Return coverage details for a policy id such as POL-100234."""
    return POLICIES.get(policy_id, {"error": f"Policy {policy_id} not found"})


def check_claim_status(claim_id: str) -> dict[str, Any]:
    """Return the current status of a claim id such as CLM-2001."""
    return CLAIMS.get(claim_id, {"error": f"Claim {claim_id} not found"})


def calculate_payout(claim_amount: float, deductible: float) -> dict[str, Any]:
    """Payout = claim amount minus deductible (never below zero). An ESTIMATE, not an approval."""
    payout = max(0.0, float(claim_amount) - float(deductible))
    return {"claim_amount": claim_amount, "deductible": deductible, "estimated_payout": payout}


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_policy": lookup_policy,
    "check_claim_status": check_claim_status,
    "calculate_payout": calculate_payout,
}

# OpenAI function-calling schemas (used in LIVE mode)
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "lookup_policy", "description": "Look up coverage details for a policy id like POL-100234.",
        "parameters": {"type": "object", "properties": {"policy_id": {"type": "string"}}, "required": ["policy_id"]}}},
    {"type": "function", "function": {
        "name": "check_claim_status", "description": "Get the status of a claim id like CLM-2001.",
        "parameters": {"type": "object", "properties": {"claim_id": {"type": "string"}}, "required": ["claim_id"]}}},
    {"type": "function", "function": {
        "name": "calculate_payout", "description": "Estimate payout = claim amount minus deductible.",
        "parameters": {"type": "object", "properties": {
            "claim_amount": {"type": "number"}, "deductible": {"type": "number"}},
            "required": ["claim_amount", "deductible"]}}},
]
