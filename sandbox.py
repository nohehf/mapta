#!/usr/bin/env python3
"""
Docker-based sandbox implementation for the security scanner.

This module provides a sandbox environment using Docker containers for safe execution
of commands and Python code during vulnerability scanning.
"""

import os
import subprocess
import time
import logging
from typing import Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result object returned by command execution."""

    stdout: str
    stderr: str
    exit_code: int


class DockerSandbox:
    """
    Docker-based sandbox implementation.

    Provides isolated execution environment for commands and file operations
    during security scanning.
    """

    def __init__(
        self, image_name: str = "mapta-sandbox", container_name: Optional[str] = None
    ):
        """
        Initialize the Docker sandbox.

        Args:
            image_name: Docker image to use for the sandbox
            container_name: Optional container name (auto-generated if not provided)
        """
        self.image_name = image_name
        self.container_name = (
            container_name or f"sandbox_{int(time.time())}_{os.urandom(4).hex()}"
        )
        self.container_id: Optional[str] = None
        self.working_dir = "/home/user"
        self.timeout_ms = 120000  # Default 2 minutes

        # Initialize nested classes
        self.files = self._Files(self)
        self.commands = self._Commands(self)

        logger.info(f"Initializing Docker sandbox with image: {image_name}")

    def _ensure_container_running(self) -> None:
        """Ensure the Docker container is running."""
        if self.container_id is None:
            # Start the container
            cmd = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                self.container_name,
                # "--network",
                # "none",  # Isolated network
                "--memory",
                "1024m",  # Memory limit
                "--cpus",
                "1",  # CPU limit
                "--read-only",  # Read-only root filesystem
                "--tmpfs",
                "/tmp:rw,mode=1777",  # Temporary filesystem writable for all users
                "--tmpfs",
                f"{self.working_dir}:rw,uid=1000,gid=1000,mode=0775",  # Working directory as tmpfs owned by 'user'
                self.image_name,
                "sleep",
                "infinity",  # Keep container running
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to start container: {result.stderr}")

                self.container_id = result.stdout.strip()
                logger.info(f"Started container: {self.container_id}")

                # Wait a moment for container to be ready
                time.sleep(1)

                # Set up the working directory
                self._run_docker_command(["mkdir", "-p", self.working_dir])

            except subprocess.TimeoutExpired:
                raise RuntimeError("Timeout starting Docker container")
            except Exception as e:
                raise RuntimeError(f"Failed to start container: {e}")

    def _run_docker_command(
        self, command: list, timeout: Optional[int] = None
    ) -> CommandResult:
        """Run a command inside the Docker container."""

        self._ensure_container_running()
        if self.container_id is None:
            raise RuntimeError("Container not started")

        if timeout is None:
            timeout = self.timeout_ms // 1000  # Convert ms to seconds

        docker_cmd = ["docker", "exec", self.container_id] + command

        try:
            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=timeout
            )

            return CommandResult(
                stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {timeout} seconds")
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=-1,
            )

    def set_timeout(self, timeout: int) -> None:
        """
        Set the timeout for command execution.

        Args:
            timeout: Timeout in milliseconds
        """
        self.timeout_ms = timeout
        logger.info(f"Set sandbox timeout to {timeout}ms")

    def kill(self) -> None:
        """Kill the sandbox container."""
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "kill", self.container_id],
                    capture_output=True,
                    timeout=10,
                )
                logger.info(f"Killed container: {self.container_id}")
            except Exception as e:
                logger.warning(f"Failed to kill container: {e}")
            finally:
                self.container_id = None

    class _Files:
        """File operations within the sandbox."""

        def __init__(self, sandbox: "DockerSandbox"):
            self.sandbox = sandbox

        def write(self, path: str, content: str) -> None:
            """
            Write content to a file in the sandbox.

            Args:
                path: Path to the file (relative to sandbox working directory)
                content: Content to write
            """
            # Ensure container is up
            self.sandbox._ensure_container_running()

            # Determine absolute path inside container
            container_path = (
                path if path.startswith("/") else f"{self.sandbox.working_dir}/{path}"
            )

            # Ensure directory exists
            dir_path = os.path.dirname(container_path)
            if dir_path and dir_path != "/":
                self.sandbox._run_docker_command(["mkdir", "-p", dir_path])

            # Stream content into the container using docker exec with stdin to avoid
            # docker cp failures on read-only rootfs; tmpfs at working_dir is writable
            if not self.sandbox.container_id:
                raise RuntimeError("Container not started")

            exec_cmd = [
                "docker",
                "exec",
                "-i",
                "-u",
                "user",
                self.sandbox.container_id,
                "sh",
                "-c",
                f"cat > '{container_path}'",
            ]
            result = subprocess.run(
                exec_cmd,
                input=content,
                text=True,
                capture_output=True,
                timeout=15,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to write file in container: {result.stderr}"
                )

            logger.info(f"Wrote {len(content)} bytes to {container_path}")

    class _Commands:
        """Command execution within the sandbox."""

        def __init__(self, sandbox: "DockerSandbox"):
            self.sandbox = sandbox

        def run(
            self,
            command: str,
            timeout: Optional[int] = None,
            user: Optional[str] = None,
        ) -> CommandResult:
            """
            Run a command in the sandbox.

            Args:
                command: Shell command to execute
                timeout: Timeout in seconds (optional)
                user: User to run as (optional, defaults to root)

            Returns:
                CommandResult with stdout, stderr, and exit code
            """
            self.sandbox._ensure_container_running()

            # Change to working directory and run command
            full_command = ["sh", "-c", f"cd {self.sandbox.working_dir} && {command}"]

            # Convert timeout from seconds to milliseconds for internal use
            if timeout:
                timeout_ms = timeout * 1000
                original_timeout = self.sandbox.timeout_ms
                self.sandbox.set_timeout(timeout_ms)
                try:
                    result = self.sandbox._run_docker_command(full_command, timeout)
                finally:
                    self.sandbox.set_timeout(original_timeout)
            else:
                result = self.sandbox._run_docker_command(full_command)

            logger.info(
                f"Executed command: {command[:50]}... (exit code: {result.exit_code})"
            )

            return result


def create_sandbox() -> DockerSandbox:
    """
    Factory function to create a new sandbox instance.

    Returns:
        DockerSandbox: A new sandbox instance
    """
    return DockerSandbox()


# For backwards compatibility and easier testing
def create_sandbox_instance(image_name: str = "mapta-sandbox") -> DockerSandbox:
    """
    Create a sandbox instance with a specific Docker image.

    Args:
        image_name: Docker image to use

    Returns:
        DockerSandbox: Configured sandbox instance
    """
    return DockerSandbox(image_name=image_name)


if __name__ == "__main__":
    # Quick test of the sandbox
    print("Testing Docker sandbox...")

    try:
        sandbox = create_sandbox()
        print("Sandbox created successfully")

        # Test file writing
        sandbox.files.write("test.txt", "Hello from sandbox!")
        print("File written successfully")

        # Test command execution
        result = sandbox.commands.run("ls -la")
        print(f"Command executed: exit code {result.exit_code}")
        print(f"Output: {result.stdout[:100]}...")

        # Clean up
        sandbox.kill()
        print("Sandbox cleaned up")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
