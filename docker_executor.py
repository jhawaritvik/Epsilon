"""
Docker Executor Module for Epsilon

This module provides Docker-based code execution for the Epsilon research engine.
It allows experiment code to run in isolated containers for security and reproducibility.

Usage:
    from docker_executor import DockerExecutor, is_docker_available
    
    if is_docker_available():
        executor = DockerExecutor()
        result = executor.execute_code(code, timeout=300, experiment_dir="/path/to/dir")
"""

import subprocess
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_IMAGE = os.environ.get("DOCKER_IMAGE", "epsilon-executor:latest")
DOCKERFILE_PATH = Path(__file__).parent / "docker" / "Dockerfile.execution"


@dataclass
class ExecutionResult:
    """Result of a Docker execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error_message: Optional[str] = None


# =============================================================================
# Docker Availability Check
# =============================================================================

def is_docker_available() -> bool:
    """
    Check if Docker is installed and running.
    
    Returns:
        True if Docker is available and the daemon is running, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Docker not available: {e}")
        return False


def is_image_available(image_name: str = DEFAULT_IMAGE) -> bool:
    """
    Check if the Docker image exists locally.
    
    Args:
        image_name: Name of the Docker image to check.
        
    Returns:
        True if image exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Image check failed: {e}")
        return False


# =============================================================================
# Docker Executor Class
# =============================================================================

class DockerExecutor:
    """
    Executes Python code inside Docker containers.
    
    This executor provides:
    - Isolated execution environment
    - Automatic volume mounting for artifact retrieval
    - Timeout enforcement at container level
    - Clean container teardown
    """
    
    def __init__(self, image_name: str = DEFAULT_IMAGE):
        """
        Initialize the Docker executor.
        
        Args:
            image_name: Docker image to use for execution.
        """
        self.image_name = image_name
        self._check_prerequisites()
    
    def _check_prerequisites(self) -> None:
        """Verify Docker is available and image exists."""
        if not is_docker_available():
            raise RuntimeError(
                "Docker is not available. Please ensure Docker is installed and running."
            )
        
        if not is_image_available(self.image_name):
            logger.warning(
                f"Docker image '{self.image_name}' not found. "
                "Run 'docker build -f docker/Dockerfile.execution -t epsilon-executor:latest .' to build it."
            )
    
    def build_image(self, force: bool = False) -> bool:
        """
        Build the Docker execution image.
        
        Args:
            force: If True, rebuild even if image exists.
            
        Returns:
            True if build succeeded, False otherwise.
        """
        if not force and is_image_available(self.image_name):
            logger.info(f"Image '{self.image_name}' already exists.")
            return True
        
        logger.info(f"Building Docker image '{self.image_name}'...")
        
        repo_root = Path(__file__).parent
        dockerfile = repo_root / "docker" / "Dockerfile.execution"
        
        if not dockerfile.exists():
            logger.error(f"Dockerfile not found: {dockerfile}")
            return False
        
        try:
            result = subprocess.run(
                [
                    "docker", "build",
                    "-f", str(dockerfile),
                    "-t", self.image_name,
                    str(repo_root)
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for build
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully built image '{self.image_name}'")
                return True
            else:
                logger.error(f"Docker build failed:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Docker build timed out after 10 minutes")
            return False
        except Exception as e:
            logger.error(f"Docker build error: {e}")
            return False
    
    def execute_code(
        self,
        code: str,
        experiment_dir: str,
        timeout: int = 300,
        script_name: str = "run_experiment.py"
    ) -> ExecutionResult:
        """
        Execute Python code inside a Docker container.
        
        Args:
            code: Python code to execute.
            experiment_dir: Host directory for experiment artifacts (will be mounted).
            timeout: Maximum execution time in seconds.
            script_name: Name for the script file.
            
        Returns:
            ExecutionResult with execution details.
        """
        experiment_path = Path(experiment_dir)
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Write code to the experiment directory
        script_path = experiment_path / script_name
        script_path.write_text(code, encoding="utf-8")
        
        # Generate unique container name
        import uuid
        container_name = f"epsilon-exec-{uuid.uuid4().hex[:8]}"
        
        # Build docker run command
        # Mount the experiment directory as /workspace/experiments
        # The script will be available inside the container
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after exit
            "--name", container_name,
            "-v", f"{experiment_path.absolute()}:/workspace/experiments:rw",
            "-w", "/workspace/experiments",
            "--network", "none",  # No network access for security
            "--memory", "4g",  # Memory limit
            "--cpus", "2",  # CPU limit
            self.image_name,
            "python", script_name
        ]
        
        logger.info(f"Executing in Docker container: {container_name}")
        logger.debug(f"Command: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(experiment_path)
            )
            
            # Write execution log
            log_content = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            (experiment_path / "execution.log").write_text(log_content, encoding="utf-8")
            
            return ExecutionResult(
                success=(result.returncode == 0),
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr
            )
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Container {container_name} timed out after {timeout}s")
            
            # Try to stop and remove the container
            self._cleanup_container(container_name)
            
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
                error_message=f"Execution timed out after {timeout} seconds"
            )
            
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            self._cleanup_container(container_name)
            
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                error_message=str(e)
            )
    
    def _cleanup_container(self, container_name: str) -> None:
        """
        Stop and remove a container if it exists.
        
        Args:
            container_name: Name of the container to cleanup.
        """
        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                timeout=30
            )
            # Remove container (in case --rm didn't work)
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10
            )
        except Exception as e:
            logger.debug(f"Container cleanup error (may be normal): {e}")
    
    def install_package(self, package_name: str) -> str:
        """
        Note: Package installation in Docker happens at image build time.
        Runtime installation is not recommended for reproducibility.
        
        This method logs a warning and suggests rebuilding the image.
        
        Args:
            package_name: Name of the package.
            
        Returns:
            Message indicating how to add the package.
        """
        message = (
            f"Package '{package_name}' is not installed in the Docker image. "
            f"To add it:\n"
            f"1. Add '{package_name}' to requirements.txt\n"
            f"2. Rebuild the image: docker build -f docker/Dockerfile.execution -t {self.image_name} ."
        )
        logger.warning(message)
        return message


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_in_docker(
    code: str,
    experiment_dir: str,
    timeout: int = 300,
    image_name: str = DEFAULT_IMAGE
) -> Tuple[bool, str]:
    """
    Convenience function to execute code in Docker.
    
    Args:
        code: Python code to execute.
        experiment_dir: Directory for artifacts.
        timeout: Execution timeout in seconds.
        image_name: Docker image to use.
        
    Returns:
        Tuple of (success: bool, message: str).
    """
    try:
        executor = DockerExecutor(image_name)
        result = executor.execute_code(code, experiment_dir, timeout)
        
        if result.success:
            return True, "Execution successful.\nLog saved to execution.log"
        elif result.timed_out:
            return False, "Execution timed out."
        else:
            return False, (
                f"Execution failed (Return Code {result.return_code}).\n\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"STDOUT:\n{result.stdout}"
            )
    except Exception as e:
        return False, f"Docker execution error: {e}"


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Executor for Epsilon")
    parser.add_argument("--check", action="store_true", help="Check Docker availability")
    parser.add_argument("--build", action="store_true", help="Build the Docker image")
    parser.add_argument("--test", action="store_true", help="Run a test execution")
    
    args = parser.parse_args()
    
    if args.check:
        available = is_docker_available()
        print(f"Docker available: {available}")
        if available:
            image_exists = is_image_available()
            print(f"Epsilon image exists: {image_exists}")
    
    elif args.build:
        executor = DockerExecutor()
        success = executor.build_image(force=True)
        print(f"Build {'succeeded' if success else 'failed'}")
    
    elif args.test:
        test_code = """
import json
print("Hello from Docker!")
result = {"status": "success", "message": "Docker execution works!"}
with open("raw_results.json", "w") as f:
    json.dump(result, f)
print("Test completed.")
"""
        test_dir = Path(__file__).parent / "experiments" / "docker_test"
        success, message = execute_in_docker(test_code, str(test_dir))
        print(f"Success: {success}")
        print(f"Message: {message}")
    
    else:
        parser.print_help()
