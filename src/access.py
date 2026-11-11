"""User access model.

thelook has no ownership concept, so access is configuration: user -> brands / departments.
A product is in a user's scope if it matches ANY listed brand OR ANY listed department.
Unknown users and users with an empty scope are denied (fail closed).
In production this comes from signed identity claims, never from chat input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class UnknownUserError(KeyError):
    """No access entry for this user -> deny."""


@dataclass(frozen=True)
class UserAccess:
    user_id: str
    brands: tuple[str, ...] = ()
    departments: tuple[str, ...] = ()
    unrestricted: bool = False

    @property
    def has_scope(self) -> bool:
        return self.unrestricted or bool(self.brands or self.departments)

    def describe(self) -> str:
        if self.unrestricted:
            return "all products"
        parts = []
        if self.brands:
            parts.append("brands " + ", ".join(self.brands))
        if self.departments:
            parts.append("departments " + ", ".join(self.departments))
        return "products in " + " or ".join(parts)


def load_access(user_id: str, path: str | Path = "config/user_access.yaml") -> UserAccess:
    data = yaml.safe_load(Path(path).read_text()) or {}
    entry = (data.get("users") or {}).get(user_id)
    if entry is None:
        raise UnknownUserError(user_id)
    return UserAccess(
        user_id=user_id,
        brands=tuple(entry.get("brands") or ()),
        departments=tuple(entry.get("departments") or ()),
        unrestricted=bool(entry.get("unrestricted", False)),
    )
