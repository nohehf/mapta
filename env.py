"""Environment configuration and setup for the security scanner.

This module handles all environment-based configuration for the agents library,
including OpenAI/Azure OpenAI client setup and tracing configuration.

Key Features:
- Automatic Azure OpenAI vs OpenAI detection
- Automatic tracing disable for Azure to prevent API conflicts
- Manual tracing control via OPENAI_AGENTS_TRACING_DISABLED environment variable
- Centralized system prompt configuration

Environment Variables:
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint (enables Azure mode)
- AZURE_OPENAI_API_KEY: Azure OpenAI API key
- AZURE_API_VERSION: Azure OpenAI API version
- AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name
- OPENAI_API_KEY: Regular OpenAI API key
- OPENAI_AGENTS_TRACING_DISABLED: Manual tracing control (true/false)
- SANDBOX_FACTORY: Sandbox factory configuration
- SYSTEM_PROMPT: Main agent system prompt
- SANDBOX_SYSTEM_PROMPT: Sandbox agent system prompt
- VALIDATOR_SYSTEM_PROMPT: Validator agent system prompt
"""

import os
from typing import Optional
from openai import AsyncAzureOpenAI, AsyncOpenAI
from agents.models.openai_provider import OpenAIProvider
from agents.run import RunConfig
from prompts import SYSTEM_PROMPT, SANDBOX_SYSTEM_PROMPT, VALIDATOR_SYSTEM_PROMPT


def _mandatory(env_var_name: str) -> str:
    """Get a mandatory environment variable or raise an error."""
    value = os.getenv(env_var_name)
    if not value:
        raise ValueError(f"Environment variable {env_var_name} is not set")
    return value


def _optional(env_var_name: str) -> Optional[str]:
    """Get an optional environment variable."""
    return os.getenv(env_var_name)


def get_client() -> AsyncOpenAI | AsyncAzureOpenAI:
    """Create and return the appropriate OpenAI client based on environment variables.

    Supports both Azure OpenAI and regular OpenAI configurations.

    Returns:
        AsyncOpenAI or AsyncAzureOpenAI client configured from environment variables.

    Raises:
        ValueError: If no valid API configuration is found.
    """
    azure_endpoint = _optional("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        print(f"Using Azure OpenAI endpoint: {azure_endpoint}")
        return AsyncAzureOpenAI(
            api_key=_mandatory("AZURE_OPENAI_API_KEY"),
            api_version=_mandatory("AZURE_API_VERSION"),
            azure_endpoint=_mandatory("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=_mandatory("AZURE_OPENAI_DEPLOYMENT"),
        )

    if os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI API")
        return AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    raise ValueError("No OpenAI API key or Azure OpenAI endpoint provided")


def get_model() -> str:
    """Get the model name from environment variables.

    Returns:
        Model name string. For Azure, uses the deployment name.
        For OpenAI, defaults to "gpt-5".
    """
    azure_model = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT"
    )  # WTF WE HAVE TO PUT DEPLOYMENT HERE, NOT MODEL
    if azure_model:
        return azure_model

    return "gpt-5"


def setup_agents_config() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str, RunConfig]:
    """Set up the complete agents configuration from environment variables.

    Returns:
        Tuple of (client, model_name, run_config) ready for use with agents.
    """
    client = get_client()
    model = get_model()

    print(f"Using model: {model}")

    # Create OpenAI provider with the client created from environment
    openai_provider = OpenAIProvider(openai_client=client)

    # Configure tracing based on environment
    tracing_disabled = is_tracing_disabled()

    if tracing_disabled:
        is_azure = _optional("AZURE_OPENAI_ENDPOINT") is not None
        tracing_env = _optional("OPENAI_AGENTS_TRACING_DISABLED")

        if tracing_env is not None:
            print("Tracing manually disabled via OPENAI_AGENTS_TRACING_DISABLED")
        elif is_azure:
            print("Azure OpenAI detected - disabling tracing to prevent API conflicts")

    # Create run config with the custom provider
    run_config = RunConfig(
        model=model, model_provider=openai_provider, tracing_disabled=tracing_disabled
    )

    return client, model, run_config


def get_sandbox_factory() -> str:
    """Get the sandbox factory configuration from environment.

    Returns:
        Sandbox factory string in format "module:function".
    """
    return os.getenv("SANDBOX_FACTORY", "sandbox:create_sandbox")


def get_system_prompt() -> str:
    """Get the system prompt from environment or return default.

    Returns:
        System prompt string for the security scanner.
    """
    return os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)


def get_sandbox_system_prompt() -> str:
    """Get the sandbox system prompt from environment or return default.

    Returns:
        System prompt string for sandbox agents.
    """
    return os.getenv("SANDBOX_SYSTEM_PROMPT", SANDBOX_SYSTEM_PROMPT)


def get_validator_system_prompt() -> str:
    """Get the validator system prompt from environment or return default.

    Returns:
        System prompt string for validator agents.
    """
    return os.getenv("VALIDATOR_SYSTEM_PROMPT", VALIDATOR_SYSTEM_PROMPT)


def is_tracing_disabled() -> bool:
    """Check if tracing should be disabled based on configuration.

    Tracing is automatically disabled for Azure OpenAI to prevent API conflicts,
    but can be manually controlled via OPENAI_AGENTS_TRACING_DISABLED environment variable.

    Returns:
        True if tracing should be disabled, False otherwise.
    """
    is_azure = _optional("AZURE_OPENAI_ENDPOINT") is not None

    # Allow manual override via environment variable
    tracing_env = _optional("OPENAI_AGENTS_TRACING_DISABLED")
    if tracing_env is not None:
        return tracing_env.lower() in ("true", "1", "yes", "on")

    # Auto-disable tracing for Azure to avoid sending to OpenAI API
    return is_azure


async def test_model_warmup(
    run_config: RunConfig,
    model: str,
) -> bool:
    """Test if the model configuration is working by sending a simple warmup request.

    Returns:
        True if the model responds correctly, False otherwise.
    """
    try:

        print("🔥 Starting model warmup test...")
        print(f"   Model: {model}")
        print(f"   Tracing disabled: {run_config.tracing_disabled}")

        # Create a simple agent for testing
        from agents import Agent, Runner

        test_agent = Agent(
            name="WarmupTest",
            instructions="You are a test agent. When asked to respond with 'OK', simply respond with exactly 'OK' and nothing else.",
            tools=[],  # No tools needed for warmup
        )

        # Send a simple test message
        result = await Runner.run(
            starting_agent=test_agent,
            input="Please respond with exactly 'OK'",
            max_turns=1,
            run_config=run_config,
        )

        # Check if the response contains "OK"
        response = result.final_output.strip()
        success = "OK" in response

        if success:
            print("✅ Model warmup test PASSED")
            print(f"   Response: {response}")
        else:
            print("❌ Model warmup test FAILED")
            print("   Expected: 'OK' in response")
            print(f"   Got: {response}")

        return success

    except Exception as e:
        print(f"❌ Model warmup test ERROR: {e}")
        return False


def test_model_warmup_sync(
    run_config: RunConfig,
    model: str,
) -> bool:
    """Synchronous wrapper for the async warmup test.

    Returns:
        True if the model responds correctly, False otherwise.
    """
    import asyncio

    return asyncio.run(test_model_warmup(run_config, model))
