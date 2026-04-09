"""
core/llm_config.py — Per-agent LLM loader.

Agents never instantiate their own LLM client.
They call get_llm(agent_name) and receive a configured LLM instance.
Swapping a model or provider = one line in config.yaml. Agent code never changes.

Provider resolution (per agent):
  1. agent config has  provider: ollama  → ChatOllama(base_url=http://localhost:11434)
  2. agent config has  provider: openai  → ChatOpenAI (requires OPENAI_API_KEY)
  3. no agent-level provider             → fall back to top-level llm.provider
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import yaml
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap — load .env once at import time
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and cache config.yaml from the project root.

    Cached for the lifetime of the process — intentional.
    config.yaml is not expected to change at runtime.
    To pick up edits to config.yaml, restart the server.
    """
    config_path = _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _agent_config(agent_name: str) -> dict:
    """Return the llm config block for a single agent."""
    config = _load_config()
    agents = config["llm"]["agents"]
    if agent_name not in agents:
        valid = list(agents.keys())
        raise ValueError(
            f"Unknown agent '{agent_name}'. Valid names: {valid}"
        )
    return agents[agent_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = "http://localhost:11434"


def get_llm(agent_name: str) -> Union[ChatOpenAI, ChatOllama]:
    """
    Return a configured LLM client for the named agent.

    Provider is resolved in order:
      1. agent-level  provider  field in config.yaml
      2. top-level    llm.provider  field in config.yaml
      3. defaults to  openai

    Parameters
    ----------
    agent_name : str
        One of: chat, planner, router, librarian, data_scientist,
        synthesizer, auditor.

    Returns
    -------
    ChatOpenAI | ChatOllama
        A LangChain chat model instance configured for that agent.

    Raises
    ------
    ValueError
        If agent_name is not present in config.yaml.
    EnvironmentError
        If provider is openai and OPENAI_API_KEY is not set.
    """
    config = _load_config()
    cfg    = _agent_config(agent_name)

    provider = cfg.get("provider") or config["llm"].get("provider", "openai")

    if provider == "ollama":
        return ChatOllama(
            model=cfg["model"],
            temperature=cfg["temperature"],
            base_url=_OLLAMA_BASE_URL,
        )

    # Default: openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Isolated test — run with: python core/llm_config.py
# ---------------------------------------------------------------------------

def test_llm_config():
    from unittest.mock import patch

    config    = _load_config()
    agent_cfg = config["llm"]["agents"]

    # ── Test 1: all default (openai) agents return ChatOpenAI ────────
    agents = ["chat", "planner", "router", "librarian", "data_scientist", "synthesizer", "auditor"]
    print(f"{'Agent':<16} {'Provider':<10} {'Model':<12} {'Temperature'}")
    print("-" * 50)

    for name in agents:
        llm            = get_llm(name)
        expected_model = agent_cfg[name]["model"]
        expected_temp  = agent_cfg[name]["temperature"]

        assert isinstance(llm, ChatOpenAI), (
            f"{name}: expected ChatOpenAI, got {type(llm).__name__}"
        )
        assert llm.model_name == expected_model, (
            f"{name}: expected model '{expected_model}', got '{llm.model_name}'"
        )
        assert llm.temperature == expected_temp, (
            f"{name}: expected temperature {expected_temp}, got {llm.temperature}"
        )
        print(f"{name:<16} {'openai':<10} {llm.model_name:<12} {llm.temperature}")

    print("PASS: all 7 default agents return correctly configured ChatOpenAI instances")

    # ── Test 2: agent with provider: openai returns ChatOpenAI ───────
    fake_config_openai = {
        "llm": {
            "provider": "openai",
            "agents": {
                "synthesizer": {"model": "gpt-4o-mini", "temperature": 0.1, "provider": "openai"},
            },
        }
    }
    with patch(f"{__name__}._load_config", return_value=fake_config_openai):
        llm_openai = get_llm("synthesizer")

    assert isinstance(llm_openai, ChatOpenAI), (
        f"provider: openai must return ChatOpenAI, got {type(llm_openai).__name__}"
    )
    assert llm_openai.model_name  == "gpt-4o-mini"
    assert llm_openai.temperature == 0.1
    print("PASS: agent with provider: openai returns ChatOpenAI")

    # ── Test 3: agent with provider: ollama returns ChatOllama ───────
    fake_config_ollama = {
        "llm": {
            "provider": "openai",   # top-level is openai — agent-level must override
            "agents": {
                "synthesizer": {"model": "qwen2.5:7b", "temperature": 0.1, "provider": "ollama"},
            },
        }
    }
    with patch(f"{__name__}._load_config", return_value=fake_config_ollama):
        llm_ollama = get_llm("synthesizer")

    assert isinstance(llm_ollama, ChatOllama), (
        f"provider: ollama must return ChatOllama, got {type(llm_ollama).__name__}"
    )
    assert llm_ollama.model       == "qwen2.5:7b"
    assert llm_ollama.temperature == 0.1
    assert llm_ollama.base_url    == _OLLAMA_BASE_URL
    print(f"PASS: agent with provider: ollama returns ChatOllama at {_OLLAMA_BASE_URL}")

    # ── Test 4: agent-level provider overrides top-level provider ────
    fake_config_override = {
        "llm": {
            "provider": "ollama",   # top-level is ollama
            "agents": {
                "planner": {"model": "gpt-4o", "temperature": 0.0, "provider": "openai"},
            },
        }
    }
    with patch(f"{__name__}._load_config", return_value=fake_config_override):
        llm_override = get_llm("planner")

    assert isinstance(llm_override, ChatOpenAI), (
        "agent-level provider: openai must override top-level provider: ollama"
    )
    print("PASS: agent-level provider overrides top-level provider")

    # ── Test 5: unknown agent must raise ValueError ───────────────────
    try:
        get_llm("nonexistent_agent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: unknown agent raises ValueError")

    # ── Test 6: openai provider without API key raises EnvironmentError
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        get_llm("chat")
        assert False, "Should have raised EnvironmentError"
    except EnvironmentError:
        pass
    finally:
        if saved_key:
            os.environ["OPENAI_API_KEY"] = saved_key
    print("PASS: missing OPENAI_API_KEY raises EnvironmentError")

    print("\nPASS: all llm_config tests passed")


if __name__ == "__main__":
    test_llm_config()
