"""
LocalRuntime - executes commands directly on the host via subprocess.

No Docker sandbox. Used for FHE challenges where the Docker judge
must run directly on the host (no Docker-in-Docker).
"""

import os
import subprocess

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime import Runtime


class LocalRuntime(Runtime):
    """Runtime that executes commands directly on the host machine.

    No sandboxing - commands run as the current user via subprocess.
    """

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        self.sid = sid
        self.event_stream = event_stream
        self.config = config
        self.plugins = plugins or []
        self._env_vars: dict[str, str] = {}
        self.DEFAULT_ENV_VARS = {}

        if env_vars:
            self._env_vars.update(env_vars)

        logger.debug(f'LocalRuntime `{sid}` initialized')

    def add_env_vars(self, env_vars: dict[str, str]) -> None:
        self._env_vars.update(env_vars)

    def _get_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._env_vars)
        return env

    def run(self, action: CmdRunAction) -> Observation:
        timeout = action.timeout if action.timeout else self.config.sandbox.timeout
        try:
            result = subprocess.run(
                action.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.workspace_base,
                env=self._get_env(),
            )
            content = result.stdout
            if result.stderr:
                content += '\n' + result.stderr
            return CmdOutputObservation(
                content=content.strip(),
                command_id=0,
                command=action.command,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return CmdOutputObservation(
                content=f'TIMEOUT: Command exceeded {timeout}s',
                command_id=0,
                command=action.command,
                exit_code=-1,
            )
        except Exception as e:
            return ErrorObservation(content=f'Command failed: {e}')

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        return ErrorObservation(content='IPython not supported in LocalRuntime')

    def read(self, action: FileReadAction) -> Observation:
        try:
            path = action.path
            if not os.path.isabs(path):
                path = os.path.join(self.config.workspace_base, path)
            with open(path, 'r') as f:
                content = f.read()
            if action.start > 0 or action.end != -1:
                lines = content.splitlines()
                end = action.end if action.end != -1 else len(lines)
                content = '\n'.join(lines[action.start:end])
            return FileReadObservation(content=content, path=path)
        except Exception as e:
            return ErrorObservation(content=f'Failed to read file: {e}')

    def write(self, action: FileWriteAction) -> Observation:
        try:
            path = action.path
            if not os.path.isabs(path):
                path = os.path.join(self.config.workspace_base, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(action.content)
            return FileWriteObservation(content='', path=path)
        except Exception as e:
            return ErrorObservation(content=f'Failed to write file: {e}')

    def browse(self, action: BrowseURLAction) -> Observation:
        return ErrorObservation(content='Browse not supported in LocalRuntime')

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return ErrorObservation(content='Browse not supported in LocalRuntime')

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        # No sandbox - just copy locally if needed
        import shutil
        if recursive:
            shutil.copytree(host_src, sandbox_dest, dirs_exist_ok=True)
        else:
            shutil.copy2(host_src, sandbox_dest)

    def list_files(self, path: str | None = None) -> list[str]:
        target = path or self.config.workspace_base
        try:
            return os.listdir(target)
        except Exception:
            return []

    def close(self) -> None:
        pass
