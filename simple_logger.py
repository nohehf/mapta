"""Simple logging system using OpenAI agents hooks.

This module provides clean, integrated logging using the agents library's hook system.
All logging is automatically handled by the hooks - no manual integration required.
"""

import json
import logging
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

from agents.lifecycle import RunHooksBase
from agents.agent import Agent
from agents.items import ModelResponse, TResponseInputItem
from agents.run_context import RunContextWrapper, TContext
from agents.tool import Tool


class SecurityScannerHooks(RunHooksBase[Any, Agent]):
    """Simple hooks-based logger for the security scanner."""

    def __init__(self, base_log_dir: str = "logs", target_url: Optional[str] = None):
        """Initialize the hooks logger."""
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)

        # Create timestamped session directory
        self.session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)

        self.target_url = target_url

        # Initialize log files
        self.tool_calls_log = self.session_dir / "tool_calls.jsonl"
        self.llm_usage_log = self.session_dir / "llm_usage.jsonl"
        self.vulnerabilities_log = self.session_dir / "vulnerabilities.jsonl"

        # Setup logger
        self.logger = logging.getLogger(f"security_scanner_{self.session_id}")
        self.logger.setLevel(logging.INFO)

        # Add console handler if not exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Session stats
        self.stats = {
            "session_id": self.session_id,
            "start_time": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "total_tool_calls": 0,
            "total_llm_calls": 0,
            "total_tokens": 0,
            "total_vulnerabilities": 0,
            "estimated_cost": 0.0,
        }

        # Tool timing
        self.tool_start_times: dict[str, float] = {}

        self.logger.info(
            f"🚀 Security Scanner initialized - Session: {self.session_id}"
        )
        if target_url:
            self.logger.info(f"🎯 Target: {target_url}")
        self.logger.info(f"📁 Logs: {self.session_dir}")

    def _append_jsonl(self, file_path: Path, data: dict) -> None:
        """Append JSON line to file."""
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(data, f, default=str, ensure_ascii=False)
            f.write("\n")

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        system_prompt: Optional[str],
        input_items: list[TResponseInputItem],
    ) -> None:
        """Called just before invoking the LLM."""
        self.logger.info(f"🤖 [{agent.name}] Starting LLM call")

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        response: ModelResponse,
    ) -> None:
        """Called after LLM call returns."""
        # Extract usage data
        usage_data = getattr(response, "usage", None)
        if usage_data:
            prompt_tokens = int(getattr(usage_data, "prompt_tokens", 0))
            completion_tokens = int(getattr(usage_data, "completion_tokens", 0))
            total_tokens = int(
                getattr(usage_data, "total_tokens", prompt_tokens + completion_tokens)
            )

            # Estimate cost (rough GPT-4 pricing)
            estimated_cost = float(
                (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
            )

            # Log to file
            log_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": self.session_id,
                "agent_name": agent.name,
                "target_url": self.target_url,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            }
            self._append_jsonl(self.llm_usage_log, log_entry)

            # Update stats
            current_llm_calls = self.stats.get("total_llm_calls", 0)
            current_tokens = self.stats.get("total_tokens", 0)
            current_cost = self.stats.get("estimated_cost", 0.0)

            self.stats["total_llm_calls"] = (
                current_llm_calls if isinstance(current_llm_calls, int) else 0
            ) + 1
            self.stats["total_tokens"] = (
                current_tokens if isinstance(current_tokens, int) else 0
            ) + total_tokens
            self.stats["estimated_cost"] = (
                current_cost if isinstance(current_cost, (int, float)) else 0.0
            ) + estimated_cost

            # Console output
            self.logger.info(
                f"💰 [{agent.name}] LLM Usage: {total_tokens} tokens "
                f"({prompt_tokens}+{completion_tokens}) | Est. Cost: ${estimated_cost:.4f}"
            )

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
    ) -> None:
        """Called when tool invocation starts."""
        tool_id = f"{agent.name}:{tool.name}:{id(tool)}"
        self.tool_start_times[tool_id] = time.time()

        self.logger.info(f"🔧 [{agent.name}] TOOL START: {tool.name}")

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
        result: str,
    ) -> None:
        """Called after tool invocation."""
        tool_id = f"{agent.name}:{tool.name}:{id(tool)}"
        start_time = self.tool_start_times.pop(tool_id, time.time())
        execution_time = (time.time() - start_time) * 1000  # ms

        # Log to file
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "agent_name": agent.name,
            "tool_name": tool.name,
            "target_url": self.target_url,
            "execution_time_ms": execution_time,
            "result_length": len(result),
            "result_preview": result[:500] if result else None,
            "success": True,  # If we got here, it succeeded
        }
        self._append_jsonl(self.tool_calls_log, log_entry)

        # Update stats
        current_tool_calls = self.stats.get("total_tool_calls", 0)
        self.stats["total_tool_calls"] = (
            current_tool_calls if isinstance(current_tool_calls, int) else 0
        ) + 1

        # Console output
        result_preview = result[:100] + "..." if len(result) > 100 else result
        self.logger.info(
            f"🔧 [{agent.name}] TOOL END: {tool.name} | "
            f"Time: {execution_time:.1f}ms | Result: {result_preview}"
        )

        # Check for vulnerability alerts in tool results
        if tool.name == "send_security_alert" and result:
            try:
                # Try to parse the result to extract vulnerability data
                if "vulnerability" in result.lower():
                    self._log_vulnerability_from_result(agent.name, result)
            except Exception:
                pass  # Don't fail if we can't parse the result

    async def on_agent_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ) -> None:
        """Called when agent starts."""
        self.logger.info(f"🎯 [{agent.name}] Agent started")

    async def on_agent_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        output: Any,
    ) -> None:
        """Called when agent ends."""
        self.logger.info(f"🎯 [{agent.name}] Agent completed")

    def _log_vulnerability_from_result(self, agent_name: str, result: str) -> None:
        """Log vulnerability from tool result."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "agent_name": agent_name,
            "target_url": self.target_url,
            "raw_result": result,
        }
        self._append_jsonl(self.vulnerabilities_log, log_entry)

        current_vulns = self.stats.get("total_vulnerabilities", 0)
        self.stats["total_vulnerabilities"] = (
            current_vulns if isinstance(current_vulns, int) else 0
        ) + 1
        self.logger.warning(f"🚨 [{agent_name}] Vulnerability detected!")

    def log_vulnerability(
        self,
        vulnerability_type: str,
        severity: str,
        description: str,
        poc_code: Optional[str] = None,
        agent_name: str = "Unknown",
    ) -> None:
        """Manually log a vulnerability (for use in tools)."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "agent_name": agent_name,
            "target_url": self.target_url,
            "vulnerability_type": vulnerability_type,
            "severity": severity.upper(),
            "description": description,
            "poc_code": poc_code,
        }
        self._append_jsonl(self.vulnerabilities_log, log_entry)

        current_vulns = self.stats.get("total_vulnerabilities", 0)
        self.stats["total_vulnerabilities"] = (
            current_vulns if isinstance(current_vulns, int) else 0
        ) + 1

        # Color-coded console output
        colors = {
            "CRITICAL": "\033[1;91m",
            "HIGH": "\033[1;31m",
            "MEDIUM": "\033[1;33m",
            "LOW": "\033[1;36m",
        }
        color = colors.get(severity.upper(), "\033[1;37m")
        reset = "\033[0m"

        self.logger.warning(
            f"{color}🚨 VULNERABILITY: {vulnerability_type} | "
            f"Severity: {severity.upper()} | Target: {self.target_url}{reset}"
        )

    def save_summary(self) -> str:
        """Save session summary."""
        self.stats["end_time"] = datetime.now(UTC).isoformat()

        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, default=str)

        # Console summary
        self.logger.info("📊 SESSION SUMMARY:")
        self.logger.info(f"   Tool Calls: {self.stats['total_tool_calls']}")
        self.logger.info(f"   LLM Calls: {self.stats['total_llm_calls']}")
        self.logger.info(f"   Total Tokens: {self.stats['total_tokens']}")
        self.logger.info(f"   Est. Cost: ${self.stats['estimated_cost']:.4f}")
        self.logger.info(f"   Vulnerabilities: {self.stats['total_vulnerabilities']}")

        return str(summary_file)


# Global hooks instance
_global_hooks: Optional[SecurityScannerHooks] = None


def get_security_hooks(target_url: Optional[str] = None) -> SecurityScannerHooks:
    """Get or create the global security hooks instance."""
    global _global_hooks
    if _global_hooks is None:
        _global_hooks = SecurityScannerHooks(target_url=target_url)
    return _global_hooks


def log_vulnerability(
    vulnerability_type: str,
    severity: str,
    description: str,
    poc_code: Optional[str] = None,
    agent_name: str = "Unknown",
) -> None:
    """Convenience function to log vulnerabilities."""
    hooks = get_security_hooks()
    hooks.log_vulnerability(
        vulnerability_type, severity, description, poc_code, agent_name
    )


def save_session_summary() -> str:
    """Convenience function to save session summary."""
    hooks = get_security_hooks()
    return hooks.save_summary()
