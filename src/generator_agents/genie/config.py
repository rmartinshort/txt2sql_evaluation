"""Configuration, YAML agent allowlist, and environment loading for Genie agents."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATABRICKS_HOST = "https://dbc-89003fab-33b9.cloud.databricks.com"
DEFAULT_DATABRICKS_PROFILE = "DEFAULT"
DEFAULT_ROUTER_MODEL = "gemini-decision-1"
DEFAULT_EXECUTION_MODE = "STANDARD"
DEFAULT_AGENTS_CONFIG_PATH = Path(__file__).resolve().parent / "agents.yaml"


def get_env_value(var_name: str, *, extra_env_files: tuple[Path, ...] = ()) -> str | None:
    """Resolve an env var from process environment first, falling back to local `.env` files."""
    val = os.environ.get(var_name)
    if val:
        return val
    candidates = [
        *extra_env_files,
        Path("src/.env"),
        Path(".env"),
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
    ]
    seen: set[Path] = set()
    for env_path in candidates:
        resolved = env_path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            for raw_line in resolved.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == var_name:
                    cleaned = v.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
        except OSError:
            continue
    return None


def load_agents_yaml(path: Path) -> dict[str, Any]:
    """Load the Genie agents YAML config file (allowed agent IDs + optional defaults)."""
    if not path.exists():
        raise FileNotFoundError(f"Genie agents config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")
    return raw


def _parse_allowed_agent_ids(raw_agents: Any) -> tuple[str, ...]:
    """Normalize `agents:` YAML entries (list of ID strings or dicts with `id`/`agent_id`)."""
    if not raw_agents:
        return ()
    if not isinstance(raw_agents, list):
        raise ValueError(f"`agents` in agents.yaml must be a list, got {type(raw_agents).__name__}")
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw_agents:
        if isinstance(entry, str):
            aid = entry.strip()
        elif isinstance(entry, dict):
            aid = str(entry.get("id") or entry.get("agent_id") or entry.get("space_id") or "").strip()
        else:
            continue
        if aid and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return tuple(out)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Runtime settings for Databricks Genie Agent Mode and Gemini Decision routing."""

    databricks_host: str = DEFAULT_DATABRICKS_HOST
    databricks_profile: str = DEFAULT_DATABRICKS_PROFILE
    databricks_token: str | None = None
    gemini_api_key: str | None = None
    router_model: str = DEFAULT_ROUTER_MODEL
    execution_mode: str = DEFAULT_EXECUTION_MODE
    allowed_agent_ids: tuple[str, ...] = ()
    agents_config_path: Path | None = None
    enable_viz: bool = False
    min_route_confidence: float = 0.35
    followup_threshold: float = 0.65
    request_timeout_s: float = 180.0
    router_timeout_s: float = 30.0

    @classmethod
    def from_env(
        cls,
        *,
        agents_config_path: Path | None = None,
        allowed_agent_ids: tuple[str, ...] | list[str] | None = None,
        databricks_host: str | None = None,
        databricks_profile: str | None = None,
        databricks_token: str | None = None,
        gemini_api_key: str | None = None,
        router_model: str | None = None,
        execution_mode: str | None = None,
        enable_viz: bool = False,
        min_route_confidence: float = 0.35,
        followup_threshold: float = 0.65,
        request_timeout_s: float = 180.0,
    ) -> "OrchestratorConfig":
        """Build config from explicit overrides, env vars, `.env` files, and `agents.yaml`."""
        cfg_path = agents_config_path
        if cfg_path is None:
            env_cfg = get_env_value("GENIE_AGENTS_CONFIG")
            if env_cfg:
                cfg_path = Path(env_cfg)
            elif DEFAULT_AGENTS_CONFIG_PATH.exists():
                cfg_path = DEFAULT_AGENTS_CONFIG_PATH

        yaml_data: dict[str, Any] = {}
        if cfg_path is not None and cfg_path.exists():
            yaml_data = load_agents_yaml(cfg_path)
        elif agents_config_path is not None:
            # Explicit path was passed by caller and does not exist -> raise clearly
            load_agents_yaml(agents_config_path)

        resolved_ids = (
            tuple(allowed_agent_ids)
            if allowed_agent_ids is not None
            else _parse_allowed_agent_ids(yaml_data.get("agents"))
        )

        host = (
            databricks_host
            or get_env_value("DATABRICKS_HOST")
            or yaml_data.get("databricks_host")
            or DEFAULT_DATABRICKS_HOST
        ).rstrip("/")
        profile = (
            databricks_profile
            or get_env_value("DATABRICKS_CONFIG_PROFILE")
            or get_env_value("DATABRICKS_PROFILE")
            or yaml_data.get("databricks_profile")
            or DEFAULT_DATABRICKS_PROFILE
        )
        return cls(
            databricks_host=host,
            databricks_profile=profile,
            databricks_token=databricks_token or get_env_value("DATABRICKS_TOKEN"),
            gemini_api_key=gemini_api_key or get_env_value("GEMINI_API_KEY"),
            router_model=(
                router_model
                or get_env_value("GENIE_ROUTER_MODEL")
                or yaml_data.get("router_model")
                or DEFAULT_ROUTER_MODEL
            ),
            execution_mode=(
                execution_mode
                or get_env_value("GENIE_EXECUTION_MODE")
                or yaml_data.get("execution_mode")
                or DEFAULT_EXECUTION_MODE
            ).upper(),
            allowed_agent_ids=resolved_ids,
            agents_config_path=cfg_path,
            enable_viz=enable_viz,
            min_route_confidence=min_route_confidence,
            followup_threshold=followup_threshold,
            request_timeout_s=request_timeout_s,
        )
