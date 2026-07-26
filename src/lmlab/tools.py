from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    field_type: str
    required: bool = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: str
    fields: tuple[Field, ...]
    changes_state: bool = False


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, spec: ToolSpec, function) -> None:
        if spec.name in self._tools:
            raise ValueError("duplicate tool")
        self._tools[spec.name] = (spec, function)

    def execute(
        self,
        name: str,
        arguments: dict,
        granted_permissions: set[str],
        approved: bool = False,
    ):
        if name not in self._tools:
            raise ValueError("unknown tool")
        spec, function = self._tools[name]
        if spec.permission not in granted_permissions:
            raise PermissionError("permission denied")
        expected = {field.name for field in spec.fields}
        if set(arguments) != expected:
            raise ValueError("arguments do not match schema")
        for field in spec.fields:
            if field.field_type == "string" and not isinstance(
                arguments[field.name], str
            ):
                raise TypeError(f"{field.name} must be a string")
        if spec.changes_state and not approved:
            raise PermissionError("explicit approval required")
        return function(**arguments)


def run_agent(decide, registry: ToolRegistry, granted_permissions, max_steps=5):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state, trace = {}, []
    for step in range(max_steps):
        action = decide(state)
        trace.append({"step": step, "action": action})
        if action.get("type") == "finish":
            return {"status": "completed", "result": action.get("result"), "trace": trace}
        if action.get("type") != "tool":
            return {"status": "failed", "reason": "invalid action", "trace": trace}
        try:
            result = registry.execute(
                action["name"],
                action.get("arguments", {}),
                set(granted_permissions),
                bool(action.get("approved")),
            )
        except Exception as exc:
            return {"status": "failed", "reason": str(exc), "trace": trace}
        state = {"observation": result}
    return {"status": "failed", "reason": "step limit", "trace": trace}

