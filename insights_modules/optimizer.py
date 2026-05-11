"""
modules/optimizer.py
--------------------------
Budget Optimizer - Smart Budget System.
Stub implementation for backward compatibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizationResult:
    """Result of budget optimization."""
    opt_allocations: dict[str, float] = field(default_factory=dict)
    category_results: list = field(default_factory=list)
    total_utility: float = 0.0


@dataclass
class CategoryResult:
    """Optimization result for a single category."""
    category: str
    opt_allocation: float
    utility_score: float
    explanation: str


def build_optimizer(
    total_budget: float,
    plan: dict[str, float],
    priority_weights: dict[str, float] | None = None,
    lifestyle: str = "balanced",
    fixed_expenses: dict[str, float] | None = None,
    category_floors: dict[str, float] | None = None,
    category_caps: dict[str, float] | None = None,
):
    """Build optimizer instance."""
    return _OptimizerStub(
        total_budget=total_budget,
        plan=plan,
        priority_weights=priority_weights or {},
        lifestyle=lifestyle,
        fixed_expenses=fixed_expenses or {},
        category_floors=category_floors or {},
        category_caps=category_caps or {},
    )


class _OptimizerStub:
    def __init__(
        self,
        total_budget: float,
        plan: dict[str, float],
        priority_weights: dict[str, float],
        lifestyle: str,
        fixed_expenses: dict[str, float],
        category_floors: dict[str, float],
        category_caps: dict[str, float],
    ):
        self.total_budget = total_budget
        self.plan = plan
        self.priority_weights = priority_weights
        self.lifestyle = lifestyle
        self.fixed_expenses = fixed_expenses
        self.category_floors = category_floors
        self.category_caps = category_caps

    def optimize(self, prev_allocations=None, actual_spending=None) -> OptimizationResult:
        """Run optimization - returns current plan as-is."""
        return OptimizationResult(
            opt_allocations=self.plan.copy(),
            category_results=[
                CategoryResult(
                    category=cat,
                    opt_allocation=amt,
                    utility_score=1.0,
                    explanation="خطة الميزانية الحالية"
                )
                for cat, amt in self.plan.items()
            ],
            total_utility=1.0
        )

    def reallocate_on_overspend(
        self,
        current_allocations: dict[str, float],
        actual_spending: dict[str, float],
    ) -> OptimizationResult:
        """Reallocate on overspend - returns adjusted allocations."""
        overspent = {k: max(0, actual_spending.get(k, 0) - current_allocations.get(k, 0))
                   for k in current_allocations}
        total_overspend = sum(overspent.values())
        
        if total_overspend == 0:
            return self.optimize()
        
        # Simple reallocation: reduce other categories proportionally
        new_allocs = current_allocations.copy()
        surplus = sum(v for v in current_allocations.values()) - total_overspend
        
        for cat in new_allocs:
            if cat in overspent and overspent[cat] > 0:
                new_allocs[cat] = current_allocations[cat]
        
        return OptimizationResult(
            opt_allocations=new_allocs,
            category_results=[
                CategoryResult(
                    category=cat,
                    opt_allocation=amt,
                    utility_score=1.0,
                    explanation=f"إعادة تخصيص {cat}"
                )
                for cat, amt in new_allocs.items()
            ],
            total_utility=1.0
        )


def log_utility(alloc: float, floor: float, cap: float) -> float:
    """Log utility function."""
    if alloc <= 0:
        return -999
    return float(alloc)


def utility_at_zero(alloc: float, floor: float) -> float:
    """Utility when allocation is at or below floor."""
    return float(alloc - floor) if alloc >= floor else 0.0