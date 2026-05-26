from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np


@dataclass
class Condition:
    name: str
    enabled: bool
    scaling_function: Callable
    extraction_function: Callable
    dimension: int = 1  # Default is 1 for most conditions


class ConditionManager:
    def __init__(self):
        self._conditions = {}

    def add_condition(self, condition: Condition):
        self._conditions[condition.name] = condition

    def enable_condition(self, condition_name: str):
        if condition_name in self._conditions:
            self._conditions[condition_name].enabled = True

    def disable_condition(self, condition_name: str):
        if condition_name in self._conditions:
            self._conditions[condition_name].enabled = False

    def update_condition_dimension(self, condition_name: str, dimension: int):
        """Update the dimension of a condition based on dataset analysis"""
        if condition_name in self._conditions:
            self._conditions[condition_name].dimension = dimension
            print(f"Updated {condition_name} condition dimension to {dimension}")

    @property
    def active_conditions(self) -> List[Condition]:
        return [cond for cond in self._conditions.values() if cond.enabled]

    def get_condition_values(self, patch_info: Dict) -> List[float]:
        values = []
        for condition in self.active_conditions:
            raw_value = condition.extraction_function(patch_info)
            scaled_value = condition.scaling_function(raw_value)
            values.append(scaled_value)
        return values


def create_default_condition_manager():
    """Create condition manager with 4 meniscus tissue conditions"""
    manager = ConditionManager()

    # Fiber Volume Fraction (FVF)
    fvf_condition = Condition(
        name='fvf',
        enabled=True,
        scaling_function=lambda x: x,  # Already in [0,1]
        extraction_function=lambda x: x['fvf_value'],
        dimension=1
    )
    manager.add_condition(fvf_condition)

    # Fiber Alignment (Rose Diagram MRL)
    alignment_condition = Condition(
        name='alignment',
        enabled=True,
        scaling_function=lambda x: x,  # Already in [0,1]
        extraction_function=lambda x: x['alignment_value'],
        dimension=1
    )
    manager.add_condition(alignment_condition)

    # Mean Fiber Thickness (normalized)
    thickness_condition = Condition(
        name='thickness',
        enabled=True,
        scaling_function=lambda x: x,  # Already normalized to [0,1]
        extraction_function=lambda x: x['thickness_value'],
        dimension=1
    )
    manager.add_condition(thickness_condition)

    # Texture Complexity
    complexity_condition = Condition(
        name='complexity',
        enabled=True,
        scaling_function=lambda x: x,  # Already in [0,1]
        extraction_function=lambda x: x['complexity_value'],
        dimension=1
    )
    manager.add_condition(complexity_condition)

    return manager