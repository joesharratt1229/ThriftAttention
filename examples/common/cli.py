from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    items: list[int] = []
    for part in str(value).replace(" ", ",").split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    if not items:
        raise ValueError("expected at least one integer")
    return items


def parse_str_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one value")
    return items


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        return _load_simple_yaml(text)

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at top level in {config_path}")
    return dict(data)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    list_key: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if list_key is None:
                raise ValueError("list item without a key in simple YAML")
            data.setdefault(list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ValueError(f"cannot parse config line: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            data[key] = []
            list_key = key
        else:
            data[key] = _parse_scalar(raw_value)
            list_key = None
    return data


def _parse_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def pick(name: str, cli_value: T | None, config: dict[str, Any], default: T, caster: Callable[[Any], T] | None = None) -> T:
    value = cli_value if cli_value is not None else config.get(name, default)
    if caster is None:
        return value  # type: ignore[return-value]
    return caster(value)
