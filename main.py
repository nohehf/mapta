import asyncio
import importlib
import json
import json as json_module
import logging
import os
import threading
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import httpx
from agents import Agent, Runner, function_tool
from agents.tool import Tool

from env import (
    get_sandbox_factory,
    get_sandbox_system_prompt,
    get_system_prompt,
    get_validator_system_prompt,
    setup_agents_config,
    test_model_warmup_sync,
)
from simple_logger import get_security_hooks, log_vulnerability, save_session_summary
from simple_mcp import create_playwright_mcp_server

# --- Setup ---
# client = AsyncOpenAI()


# Set up agents configuration from environment
client, model, run_config = setup_agents_config()

# Only run model warmup when executed directly, not when imported as a module
if __name__ == "__main__":
    test_model_warmup_sync(run_config, model)


# Global sandbox configuration (sanitized for open release)
# Provide a factory via env var SANDBOX_FACTORY="your_module:create_sandbox" that returns a sandbox instance
SANDBOX_FACTORY = get_sandbox_factory()

# Thread-local storage for sandbox instances
_thread_local = threading.local()


def get_current_sandbox():
    """Get the sandbox instance for the current thread/scan."""
    return getattr(_thread_local, "sandbox", None)


def set_current_sandbox(sandbox):
    """Set the sandbox instance for the current thread/scan."""
    _thread_local.sandbox = sandbox


def create_sandbox_from_env():
    """Create a sandbox instance using a user-provided factory specified in SANDBOX_FACTORY.

    SANDBOX_FACTORY should be in the form "module_path:function_name" and must return an
    object exposing .files.write(path, content), .commands.run(cmd, timeout=..., user=...),
    and optional .set_timeout(ms) and .kill().

    Returns None if not configured.
    """
    factory_path = SANDBOX_FACTORY
    if not factory_path:
        logging.info("Sandbox factory not configured; running without a sandbox.")
        return None
    try:
        module_name, func_name = factory_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)
        sandbox = factory()
        # Optionally extend timeout if provider supports it
        if hasattr(sandbox, "set_timeout"):
            try:
                sandbox.set_timeout(timeout=12000)
            except TypeError:
                # Some providers may use milliseconds
                sandbox.set_timeout(12000)
        return sandbox
    except Exception as exc:
        logging.warning(f"Failed to create sandbox from SANDBOX_FACTORY: {exc}")
        return None


# Usage tracking
class UsageTracker:
    def __init__(self):
        self.main_agent_usage = []
        self.sandbox_agent_usage = []
        self.start_time = datetime.now(UTC)

    def log_main_agent_usage(self, usage_data, target_url=""):
        """Log usage data from main agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "main_agent",
            "usage": usage_data,
        }
        self.main_agent_usage.append(entry)
        logging.info(f"Main Agent Usage - Target: {target_url}, Usage: {usage_data}")

    def log_sandbox_agent_usage(self, usage_data, target_url=""):
        """Log usage data from sandbox agent responses."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "target_url": target_url,
            "agent_type": "sandbox_agent",
            "usage": usage_data,
        }
        self.sandbox_agent_usage.append(entry)
        logging.info(f"Sandbox Agent Usage - Target: {target_url}, Usage: {usage_data}")

    def get_summary(self):
        """Get usage summary for all agents."""
        return {
            "scan_duration": str(datetime.now(UTC) - self.start_time),
            "main_agent_calls": len(self.main_agent_usage),
            "sandbox_agent_calls": len(self.sandbox_agent_usage),
            "total_calls": len(self.main_agent_usage) + len(self.sandbox_agent_usage),
            "main_agent_usage": self.main_agent_usage,
            "sandbox_agent_usage": self.sandbox_agent_usage,
        }

    def save_to_file(self, filename_prefix=""):
        """Save usage data to JSON file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}usage_log_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2, default=str)

        logging.info(f"Usage data saved to {filename}")
        return filename


# Thread-local storage for usage trackers
def get_current_usage_tracker():
    """Get the usage tracker for the current thread/scan."""
    return getattr(_thread_local, "usage_tracker", None)


def set_current_usage_tracker(tracker):
    """Set the usage tracker for the current thread/scan."""
    _thread_local.usage_tracker = tracker


# ---- Logging helpers ----
def _truncate_text(text: Any, max_chars: int = 4000) -> str:
    try:
        s = str(text)
    except Exception:
        s = repr(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"...[+{len(s) - max_chars} chars]"


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return _truncate_text(obj)


def _log_messages(tag: str, messages: List[Any], max_chars: int = 2000) -> None:
    logging.info(f"{tag} messages count: {len(messages)}")
    for idx, m in enumerate(messages):
        role = ""
        content = None
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content")
        else:
            role = getattr(m, "role", "")
            content = getattr(m, "content", None)

        text_accum = []
        if isinstance(content, list):
            for ci in content:
                if isinstance(ci, dict) and "text" in ci:
                    text_accum.append(str(ci.get("text", "")))
                else:
                    text_accum.append(str(ci))
        else:
            if content is not None:
                text_accum.append(str(content))

        text_joined = _truncate_text("\n".join(text_accum), max_chars)
        logging.info(f"{tag} [{idx}] role={role} text=\n{text_joined}")


def _log_function_calls(
    agent_name: str, function_calls: List[Any], max_chars: int = 2000
) -> None:
    if not function_calls:
        return
    logging.info(f"[{agent_name}] Requested {len(function_calls)} tool calls")
    for fc in function_calls:
        name = getattr(fc, "name", "")
        call_id = getattr(fc, "call_id", "")
        args_raw = getattr(fc, "arguments", "")
        args_display = _truncate_text(args_raw, max_chars)
        logging.info(
            f"[{agent_name}] -> function_call name={name} call_id={call_id} args={args_display}"
        )


def _log_tool_results(
    agent_name: str, results: List[Dict[str, Any]], max_chars: int = 4000
) -> None:
    if not results:
        return
    logging.info(f"[{agent_name}] Received {len(results)} tool results")
    for res in results:
        call_id = res.get("call_id", "")
        output = res.get("output", "")
        output_display = _truncate_text(output, max_chars)
        logging.info(
            f"[{agent_name}] <- function_result call_id={call_id} output=\n{output_display}"
        )


# Simple helper to create agent with tools
def create_security_agent(system_prompt: str):
    """Create a security scanning agent with all the necessary tools."""
    # Create the Playwright MCP server
    playwright_server = create_playwright_mcp_server()
    

    return Agent(
        name="SecurityScanner",
        instructions=system_prompt,
        tools=base_tools,  # Base tools only - MCP tools are added via mcp_servers
        mcp_servers=[playwright_server]  # Add MCP server directly
    )


# In-memory store: email -> JWT token (for mail.tm API)
email_token_store: dict[str, str] = {}


@function_tool
async def get_registered_emails():
    """
    Return the list of email accounts in case you need to use them to receive emails such as account activation emails, credentials, etc.
    """
    return json_module.dumps(list(email_token_store.keys()))


@function_tool
async def list_account_messages(email: str, limit: int = 50):
    """
    List recent messages for the given email account.
    Returns JSON list: [{id, subject, from, intro, seen, createdAt}]

    Args:
        email: The email account to fetch messages for
        limit: Maximum number of messages to return (default: 50)
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get("https://api.mail.tm/messages", headers=headers)
            if resp.status_code != 200:
                return f"Failed to fetch messages. Status: {resp.status_code}, Response: {resp.text}"
            data = resp.json()
            messages = data.get("hydra:member", [])
            items = []
            for m in messages[:limit]:
                sender = m.get("from") or {}
                items.append(
                    {
                        "id": m.get("id"),
                        "subject": m.get("subject"),
                        "from": sender.get("address") or sender.get("name") or "",
                        "intro": m.get("intro", ""),
                        "seen": m.get("seen", False),
                        "createdAt": m.get("createdAt", ""),
                    }
                )
            return json_module.dumps(items)
    except Exception as e:
        return f"Request failed: {e}"


@function_tool
async def get_message_by_id(email: str, message_id: str):
    """
    Fetch a specific message by id for the given email account using its stored JWT.
    Returns JSON: {id, subject, from, text, html}

    Args:
        email: The email account to fetch the message from
        message_id: The ID of the message to fetch
    """
    jwt = email_token_store.get(email)
    if not jwt:
        return f"No JWT token stored for {email}. Call set_email_jwt_token(email, jwt_token) first."

    headers = {"Authorization": f"Bearer {jwt}"}
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"https://api.mail.tm/messages/{message_id}", headers=headers
            )
            if resp.status_code != 200:
                return f"Failed to fetch message. Status: {resp.status_code}, Response: {resp.text}"
            msg = resp.json()
            sender = msg.get("from") or {}
            result = {
                "id": msg.get("id"),
                "subject": msg.get("subject"),
                "from": sender.get("address") or sender.get("name") or "",
                "text": msg.get("text", ""),
                "html": msg.get("html", ""),
            }
            return json_module.dumps(result)
    except Exception as e:
        return f"Request failed: {e}"


@function_tool(name_override="send_security_alert")
async def send_security_alert(
    vulnerability_type: str,
    severity: str,
    target_url: str,
    description: str,
    evidence: Optional[str] = None,
    recommendation: Optional[str] = None,
    thread_ts: Optional[str] = None,
):
    """
    Display a security vulnerability alert to the console with rich formatting.

    Args:
        vulnerability_type: Type of vulnerability (e.g., "XSS", "SQL Injection", "IDOR")
        severity: Severity level ("Critical", "High", "Medium", "Low", "Info")
        target_url: The affected URL or endpoint
        description: Detailed description of the vulnerability
        evidence: Optional proof-of-concept or evidence details
        recommendation: Optional remediation recommendation
        thread_ts: Optional thread timestamp (ignored in console mode)
    """

    # Severity styling mapping
    severity_styles = {
        "Critical": {
            "emoji": "🚨",
            "border": "█",
            "color": "🔴 CRITICAL",
            "priority": 5,
        },
        "High": {
            "emoji": "⚠️",
            "border": "!",
            "color": "🟠 HIGH",
            "priority": 4,
        },
        "Medium": {
            "emoji": "⚡",
            "border": "~",
            "color": "🟡 MEDIUM",
            "priority": 3,
        },
        "Low": {
            "emoji": "📝",
            "border": "-",
            "color": "🟢 LOW",
            "priority": 2,
        },
        "Info": {
            "emoji": "ℹ️",
            "border": "·",
            "color": "🔵 INFO",
            "priority": 1,
        },
    }

    style = severity_styles.get(severity, severity_styles["Info"])
    emoji = style["emoji"]
    border_char = style["border"]
    color_label = style["color"]

    # Create a prominent alert box
    width = 90
    border = border_char * width  # type: ignore

    # Format target URL (remove protocol for cleaner display)
    display_url = target_url.replace("https://", "").replace("http://", "")

    # Build the alert display
    alert_lines = [
        "",
        f"\033[1;31m{border}\033[0m",  # Red border
        f"\033[1;31m{border_char}\033[0m {emoji} {color_label} VULNERABILITY ALERT {emoji} \033[1;31m{border_char}\033[0m".center(
            width
        ),
        f"\033[1;31m{border}\033[0m",
        "",
        f"\033[1;33m🎯 Vulnerability:\033[0m {vulnerability_type}",
        f"\033[1;33m🎯 Target:\033[0m {display_url}",
        f"\033[1;33m📊 Severity:\033[0m {severity.upper()}",
        "",
        "\033[1;36m📋 Description:\033[0m",
        f"   {description}",
        "",
    ]

    # Add evidence if provided
    if evidence:
        alert_lines.extend(
            [
                "\033[1;35m🔍 Evidence/PoC:\033[0m",
                f"   {evidence[:200]}{'...' if len(evidence) > 200 else ''}",
                "",
            ]
        )

    # Add recommendation if provided
    if recommendation:
        alert_lines.extend(
            [
                "\033[1;32m💡 Recommendation:\033[0m",
                f"   {recommendation}",
                "",
            ]
        )

    # Add timestamp
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    alert_lines.extend(
        [
            f"\033[1;90m⏰ Detected at: {timestamp}\033[0m",
            f"\033[1;31m{border}\033[0m",
            "",
        ]
    )

    # Print the alert with colors
    for line in alert_lines:
        print(line)

    # Also print a simplified version for logging
    simple_alert = f"[{severity.upper()}] {vulnerability_type} vulnerability detected at {target_url}"
    print(f"\033[1;31m📢 ALERT: {simple_alert}\033[0m")
    print()

    # Log to file with structured format
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Security Alert - Type: {vulnerability_type}, Severity: {severity}, "
        f"Target: {target_url}, Description: {description[:100]}..."
    )

    # Simple vulnerability logging
    log_vulnerability(
        vulnerability_type=vulnerability_type,
        severity=severity,
        description=description,
        poc_code=evidence,
        agent_name="SecurityScanner",
    )

    # Return structured response for compatibility
    return json_module.dumps(
        {
            "success": True,
            "message": "Security alert displayed to console",
            "alert_type": "vulnerability",
            "vulnerability_type": vulnerability_type,
            "severity": severity,
            "target_url": target_url,
            "description": description,
            "evidence": evidence,
            "recommendation": recommendation,
            "timestamp": timestamp,
        }
    )


@function_tool(name_override="send_scan_summary")
async def send_scan_summary(
    target_url: str,
    total_findings: int,
    critical_count: int = 0,
    high_count: int = 0,
    medium_count: int = 0,
    low_count: int = 0,
    scan_duration: Optional[str] = None,
):
    """
    Display a pretty summary of the security scan to the console.

    Args:
        target_url: The target that was scanned
        total_findings: Total number of vulnerabilities found
        critical_count: Number of critical severity findings
        high_count: Number of high severity findings
        medium_count: Number of medium severity findings
        low_count: Number of low severity findings
        scan_duration: Optional duration of the scan
    """

    # Determine overall status
    if critical_count > 0:
        status_emoji = "🔴"
        status_text = "CRITICAL ISSUES FOUND"
        border_char = "!"
    elif high_count > 0:
        status_emoji = "🟠"
        status_text = "HIGH RISK ISSUES FOUND"
        border_char = "!"
    elif medium_count > 0:
        status_emoji = "🟡"
        status_text = "MEDIUM RISK ISSUES FOUND"
        border_char = "~"
    elif low_count > 0:
        status_emoji = "🟢"
        status_text = "LOW RISK ISSUES FOUND"
        border_char = "-"
    else:
        status_emoji = "✅"
        status_text = "NO ISSUES FOUND"
        border_char = "="

    # Create a nice border
    width = 80
    border = border_char * width

    # Format target URL (remove protocol for cleaner display)
    display_url = target_url.replace("https://", "").replace("http://", "")

    # Build the summary
    summary_lines = [
        "",
        border,
        f"{status_emoji}  SECURITY SCAN SUMMARY  {status_emoji}".center(width),
        border,
        "",
        f"🎯 Target: {display_url}",
        f"📊 Status: {status_text}",
        f"🔍 Total Findings: {total_findings}",
        "",
    ]

    # Add findings breakdown if any exist
    if total_findings > 0:
        summary_lines.append("📋 FINDINGS BREAKDOWN:")
        summary_lines.append("")

        if critical_count > 0:
            summary_lines.append(f"  🚨 Critical: {critical_count}")
        if high_count > 0:
            summary_lines.append(f"  ⚠️  High:     {high_count}")
        if medium_count > 0:
            summary_lines.append(f"  ⚡  Medium:   {medium_count}")
        if low_count > 0:
            summary_lines.append(f"  📝 Low:      {low_count}")

        summary_lines.append("")

    # Add scan duration if provided
    if scan_duration:
        summary_lines.append(f"⏱️  Scan Duration: {scan_duration}")
        summary_lines.append(
            f"📅 Completed: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    summary_lines.append(border)
    summary_lines.append("")

    # Print the summary
    for line in summary_lines:
        print(line)

    # Log to file as well
    logger = logging.getLogger(__name__)
    logger.info(
        f"Scan Summary - Target: {target_url}, Status: {status_text}, Findings: {total_findings}"
    )

    return json_module.dumps(
        {
            "success": True,
            "message": "Scan summary displayed to console",
            "target": target_url,
            "status": status_text,
            "total_findings": total_findings,
            "breakdown": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
            "scan_duration": scan_duration,
        }
    )


async def _run_sandbox_agent_impl(input: str, max_rounds: int = 100):
    """
    Nested agent loop that uses only sandbox execution tools to fulfill the provided instruction.
    Returns the final textual response when the model stops requesting tools or when max_rounds is hit.

    Args:
        instruction: The instruction for the sandbox agent to execute
        max_rounds: Maximum number of execution rounds (default: 100)
    """
    sandbox_system_prompt = get_sandbox_system_prompt()

    logging.info(f"[sandbox_agent] Starting with max_rounds={max_rounds}")
    logging.info(f"[sandbox_agent] Input length: {len(input)} characters")

    # Create sandbox agent with only the low-level tools
    sandbox_tools: List[Tool] = [sandbox_run_command, sandbox_run_python]
    sandbox_agent = Agent(
        name="SandboxAgent",
        instructions=sandbox_system_prompt,
        tools=sandbox_tools,
    )

    logging.info(f"[sandbox_agent] Created agent with {len(sandbox_tools)} tools")

    try:
        # Use the SDK's Runner with hooks
        hooks = get_security_hooks(getattr(_thread_local, "current_target_url", None))
        result = await Runner.run(
            starting_agent=sandbox_agent,
            input=input,
            max_turns=max_rounds,
            run_config=run_config,
            hooks=hooks,
        )

        logging.info("[sandbox_agent] Completed successfully")
        logging.info(
            f"[sandbox_agent] Output length: {len(result.final_output) if result.final_output else 0} characters"
        )

        # Log sandbox agent usage
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(result, "usage"):
            target_url = getattr(_thread_local, "current_target_url", "")
            usage_tracker.log_sandbox_agent_usage(result.usage, target_url)

        return result.final_output

    except Exception as e:
        logging.error(f"[sandbox_agent] Error during execution: {e}")
        return f"[sandbox_agent] Error: {e}"


async def _run_validator_agent_impl(input: str, max_rounds: int = 50):
    """
    Agent loop specialized for validating Proofs-of-Concept (PoCs) in the sandbox.
    Use only sandbox tools, keep outputs concise, and return a clear verdict.

    Args:
        instruction: Validation instruction that includes the PoC and expected outcome
        max_rounds: Maximum number of execution rounds (default: 50)
    """
    validator_system_prompt = get_validator_system_prompt()

    logging.info(f"[validator_agent] Starting with max_rounds={max_rounds}")
    logging.info(f"[validator_agent] Input length: {len(input)} characters")

    # Create validator agent with only the low-level tools
    validator_tools: List[Tool] = [sandbox_run_command, sandbox_run_python]
    validator_agent = Agent(
        name="ValidatorAgent",
        instructions=validator_system_prompt,
        tools=validator_tools,
    )

    logging.info(f"[validator_agent] Created agent with {len(validator_tools)} tools")

    try:
        # Use the SDK's Runner with hooks
        hooks = get_security_hooks(getattr(_thread_local, "current_target_url", None))
        result = await Runner.run(
            starting_agent=validator_agent,
            input=input,
            max_turns=max_rounds,
            run_config=run_config,
            hooks=hooks,
        )

        logging.info("[validator_agent] Completed successfully")
        logging.info(
            f"[validator_agent] Output length: {len(result.final_output) if result.final_output else 0} characters"
        )

        # Reuse sandbox usage tracker for validator agent
        usage_tracker = get_current_usage_tracker()
        if usage_tracker and hasattr(result, "usage"):
            target_url = getattr(_thread_local, "current_target_url", "")
            usage_tracker.log_sandbox_agent_usage(result.usage, target_url)

        return result.final_output

    except Exception as e:
        logging.error(f"[validator_agent] Error during execution: {e}")
        return f"[validator_agent] Error: {e}"


# Create decorated versions for the tool system
@function_tool(name_override="sandbox_agent")
async def run_sandbox_agent(input: str, max_rounds: int = 100):
    """
    Nested agent loop that uses only sandbox execution tools to fulfill the provided instruction.
    Returns the final textual response when the model stops requesting tools or when max_rounds is hit.

    Args:
        input: The instruction for the sandbox agent to execute
        max_rounds: Maximum number of execution rounds (default: 100)
    """
    return await _run_sandbox_agent_impl(input, max_rounds)


@function_tool(name_override="validator_agent")
async def run_validator_agent(input: str, max_rounds: int = 50):
    """
    Agent loop specialized for validating Proofs-of-Concept (PoCs) in the sandbox.
    Use only sandbox tools, keep outputs concise, and return a clear verdict.

    Args:
        input: Validation instruction that includes the PoC and expected outcome
        max_rounds: Maximum number of execution rounds (default: 50)
    """
    return await _run_validator_agent_impl(input, max_rounds)


@function_tool
async def sandbox_run_python(python_code: str, timeout: int = 120):
    """
    Run Python code inside a Docker sandbox and return stdout/stderr/exit code. If the output exceeds 30000 characters, output will be truncated before being returned to you.

    Args:
        python_code: Python code to execute (e.g., "print('Hello World')").
        timeout: Max seconds to wait before timing out the code execution.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running Python code: {python_code[:100]}...")
    try:
        # Get the current sandbox instance
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"

        import uuid

        # Generate a random script name
        script_name = f"temp_script_{uuid.uuid4().hex[:8]}.py"
        script_path = f"/home/user/{script_name}"

        # Write Python code to a temporary file with random name
        sbx.files.write(script_path, python_code)

        # Execute the Python script using configured sandbox
        result = sbx.commands.run(
            f"source .venv/bin/activate && python3 {script_path}",
            timeout=timeout,
            user="root",
        )

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        output = (
            f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"
        )

        # Truncate output if it exceeds 30000 characters
        if len(output) > 30000:
            output = (
                output[:30000] + "\n...[OUTPUT TRUNCATED - EXCEEDED 30000 CHARACTERS]"
            )

        return output
    except Exception as e:
        return f"Failed to run Python code in sandbox: {e}"


@function_tool
async def sandbox_run_command(command: str, timeout: int = 120):
    """
    Run a shell command inside an ephemeral sandbox and return stdout/stderr/exit code.

    Arguments:
        command: Shell command to execute (e.g., "ls -la").
        timeout: Max seconds to wait before timing out the command.

    Returns:
        A string containing exit code, stdout, and stderr.
    """

    print(f"Running command: {command}")
    try:
        # Get the current sandbox instance
        sbx = get_current_sandbox()
        if sbx is None:
            return "Error: No sandbox instance available for this scan"

        # Use the current sandbox instance
        result = sbx.commands.run(command, timeout=timeout, user="root")

        def clip_to_max_lines(text: str, max_lines: int = 100) -> str:
            if not text:
                return ""
            lines = text.splitlines()
            if len(lines) <= max_lines:
                return "\n".join(lines)
            visible = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return f"{visible}\n...[TRUNCATED {remaining} more lines]"

        stdout_raw = (
            result.stdout
            if hasattr(result, "stdout") and result.stdout is not None
            else ""
        )
        stderr_raw = (
            result.stderr
            if hasattr(result, "stderr") and result.stderr is not None
            else ""
        )
        # stdout = clip_to_max_lines(stdout_raw, 50)
        # stderr = clip_to_max_lines(stderr_raw, 50)
        exit_code = result.exit_code if hasattr(result, "exit_code") else "unknown"

        return f"Exit code: {exit_code}\n\nSTDOUT\n{stdout_raw}\n\nSTDERR\n{stderr_raw}"
    except Exception as e:
        return f"Failed to run command in sandbox: {e}"


# Collect all function tools for the agent
tools: List[Tool] = [
    sandbox_run_command,
    sandbox_run_python,
    run_sandbox_agent,
    run_validator_agent,
    get_message_by_id,
    list_account_messages,
    get_registered_emails,
    send_security_alert,
    send_scan_summary,
]


user_prompt = """i need you to come up with detailed poc for the workflow code injection vulnerability

"""


def read_targets_from_file(file_path: str) -> List[str]:
    """
    Read target URLs from a text file, one per line.
    Ignores empty lines and lines starting with #.
    """
    targets = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    targets.append(line)
        return targets
    except FileNotFoundError:
        print(f"Error: Target file '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading target file: {e}")
        return []


async def run_continuously(
    max_rounds: int = 100,
    user_prompt: str = "",
    system_prompt: str = "",
    target_url: str = "",
    sandbox_instance=None,
):
    """
    Keep prompting the model and executing any requested tool calls in parallel
    until the model stops requesting tools or the optional max_rounds is reached.

    max_rounds: 0 means unlimited; otherwise, loop up to max_rounds tool-execution rounds.
    target_url: The target URL being scanned (used for metadata)
    sandbox_instance: Specific sandbox instance to use for this scan
    """
    # Create sandbox instance if not provided
    if sandbox_instance is None:
        sandbox_instance = create_sandbox_from_env()

    # Set the sandbox for this thread/scan
    set_current_sandbox(sandbox_instance)

    # Set target URL for usage tracking
    _thread_local.current_target_url = target_url

    # Create usage tracker
    usage_tracker = get_current_usage_tracker()

    # Extract site name from URL for metadata (used for logging context)
    site_name = (
        target_url.replace("https://", "").replace("http://", "").split("/")[0]
        if target_url
        else "unknown"
    )
    logging.info(f"[main_agent] Site: {site_name}")

    try:
        logging.info(f"[main_agent] Starting security scan for {target_url}")
        logging.info(f"[main_agent] Max rounds: {max_rounds}")

        # Create agent and run using the SDK
        agent = create_security_agent(system_prompt)

        await agent.mcp_servers[0].connect()
        mcp_tools = await agent.mcp_servers[0].list_tools()
        logging.info(f"[main_agent] MCP tools: {[tool.name for tool in mcp_tools]}")
        # Log agent details
        logging.info(f"[main_agent] Created agent with {len(agent.tools)} tools")
        logging.info(f"[main_agent] Tools: {[tool.name for tool in agent.tools]}")

        # Playwright MCP server will auto-start when first tool is called

        # Use the SDK's Runner to handle everything with hooks
        hooks = get_security_hooks(target_url)
        logging.info(f"[main_agent] Starting Runner.run with max_turns={max_rounds}")
        result = await Runner.run(
            starting_agent=agent,
            input=user_prompt,
            max_turns=max_rounds,
            run_config=run_config,
            hooks=hooks,
        )

        # Log completion
        logging.info("[main_agent] Runner completed successfully")
        logging.info(
            f"[main_agent] Final output length: {len(result.final_output) if result.final_output else 0} characters"
        )

        # Log usage if available
        if usage_tracker and hasattr(result, "usage"):
            usage_tracker.log_main_agent_usage(result.usage, target_url)
            logging.info(f"[main_agent] Usage logged for target: {target_url}")

        # Print final output to console
        print(result.final_output)
        if hasattr(result, "response_id"):
            print(f"Response ID: {result.response_id}")

        return result.final_output

    except Exception as e:
        logging.error(f"[main_agent] Error during execution: {e}")
        raise

    finally:
        # Kill the sandbox when scan is done
        if sandbox_instance and hasattr(sandbox_instance, "kill"):
            logging.info("[main_agent] Cleaning up sandbox instance")
            sandbox_instance.kill()

        # Playwright MCP server cleanup is handled automatically


async def run_single_target_scan(
    target_url: str, system_prompt: str, base_user_prompt: str, max_rounds: int = 100
):
    """
    Run a security scan for a single target URL.
    Returns the scan result and saves it to a file.
    Each scan gets its own isolated sandbox instance.
    """
    print(f"Starting scan for: {target_url}")

    # Create a dedicated sandbox instance for this scan (if configured)
    sandbox_instance = create_sandbox_from_env()

    # Create usage tracker for this scan
    usage_tracker = UsageTracker()
    set_current_usage_tracker(usage_tracker)

    # Format the user prompt with the target URL
    user_prompt = base_user_prompt.format(target_url=target_url)

    try:
        # Run the scan with dedicated sandbox
        result = await run_continuously(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            target_url=target_url,
            max_rounds=max_rounds,
            sandbox_instance=sandbox_instance,
        )

        # Generate filename from target URL
        filename = (
            target_url.replace("https://", "").replace("http://", "").replace("/", "_")
            + ".md"
        )

        # Save result to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result)

        # Save usage data
        site_name = (
            target_url.replace("https://", "").replace("http://", "").split("/")[0]
        )
        usage_filename = usage_tracker.save_to_file(f"{site_name}_")

        print(f"Scan completed for {target_url} - Results saved to {filename}")
        print(f"Usage data saved to {usage_filename}")

        return {
            "target": target_url,
            "filename": filename,
            "usage_filename": usage_filename,
            "status": "completed",
            "result": result,
            "usage_summary": usage_tracker.get_summary(),
        }

    except Exception as e:
        print(f"Error scanning {target_url}: {e}")
        return {
            "target": target_url,
            "filename": None,
            "status": "error",
            "error": str(e),
        }


async def run_parallel_scans(
    targets: List[str], system_prompt: str, base_user_prompt: str, max_rounds: int = 100
):
    """
    Run security scans for multiple targets in parallel.
    """
    print(f"Starting parallel scans for {len(targets)} targets...")

    # Create tasks for all targets
    tasks = [
        run_single_target_scan(target, system_prompt, base_user_prompt, max_rounds)
        for target in targets
    ]

    # Run all scans in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    completed = 0
    errors = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            errors += 1
        elif result.get("status") == "completed":  # type: ignore
            completed += 1
        else:
            errors += 1

    print("\nScan Summary:")
    print(f"Total targets: {len(targets)}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {errors}")

    # Create overall usage summary
    total_main_calls = 0
    total_sandbox_calls = 0
    usage_files = []

    for result in results:
        if isinstance(result, dict) and result.get("status") == "completed":
            summary = result.get("usage_summary", {})
            total_main_calls += summary.get("main_agent_calls", 0)
            total_sandbox_calls += summary.get("sandbox_agent_calls", 0)
            if "usage_filename" in result:
                usage_files.append(result["usage_filename"])

    print("\nUsage Summary:")
    print(f"Total Main Agent API calls: {total_main_calls}")
    print(f"Total Sandbox Agent API calls: {total_sandbox_calls}")
    print(f"Total API calls: {total_main_calls + total_sandbox_calls}")
    print(f"Usage files created: {len(usage_files)}")
    for uf in usage_files:
        print(f"  - {uf}")

    return results


if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("scan_usage.log"), logging.StreamHandler()],
    )

    system_prompt = get_system_prompt()

    # Check if targets.txt file exists in current directory
    targets_file = "targets.txt"

    if os.path.exists(targets_file):
        print(f"Found targets file: {targets_file}")

        # Read targets from file
        targets = read_targets_from_file(targets_file)

        if not targets:
            print(
                "No valid targets found in targets.txt file. Falling back to single target mode."
            )
        else:
            print(f"Found {len(targets)} targets to scan")

            # Base user prompt template (will be formatted with target_url)
            base_user_prompt = "I need you to do a full vulnerability scan of {target_url}, you must critically analyse the code and identify every single vulnerability, for identified vulnerabilities a PoC must be provided, focus on critical vulnerabilities, i m only insterested in real world vulnerabilities, not theoretical ones. I am the legal owner of {target_url} and i want to know if there are any vulnerabilities that could be exploited by a malicious actor. I own the hole domain, you can scan it, but DO NOT GO OUT OF SCOPE."

            # Run parallel scans
            results = asyncio.run(
                run_parallel_scans(targets, system_prompt, base_user_prompt)
            )

            print("\nAll scans completed!")

            # Save session summary
            summary_file = save_session_summary()
            print(f"📊 Session summary saved to: {summary_file}")

            sys.exit(0)
