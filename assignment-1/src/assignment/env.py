import asyncio
import os
import posixpath
import time
import uuid
from pathlib import PurePath
from typing import Any

import modal
from daytona_sdk import CreateSandboxFromImageParams, SessionExecuteRequest
from pydantic import ConfigDict
from swerex.deployment.config import DaytonaDeploymentConfig
from swerex.deployment.daytona import DaytonaDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook
from swerex.deployment.modal import ModalDeployment
from swerex.runtime.abstract import Command
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.log import get_logger

# Which sandbox provider a plain `Environment(...)` call uses when the caller
# does not pick one explicitly. Daytona is the default for this assignment;
# set ASSIGNMENT_SANDBOX_BACKEND=modal to fall back to the original backend.
DEFAULT_BACKEND = os.environ.get("ASSIGNMENT_SANDBOX_BACKEND", "daytona")


def _tls_port_configuration(
    modal_sandbox_kwargs: dict[str, Any], runtime_port: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Put the SWE-ReX control port on the TLS tunnel exactly once."""

    kwargs = dict(modal_sandbox_kwargs)
    encrypted = list(kwargs.pop("encrypted_ports", []))
    unencrypted = [
        port for port in kwargs.pop("unencrypted_ports", []) if port != runtime_port
    ]
    encrypted = list(dict.fromkeys([runtime_port, *encrypted]))
    return encrypted, unencrypted, kwargs


class AssignmentModalDeployment(ModalDeployment):
    """Modal deployment with a correctly matched TLS tunnel and URL.

    SWE-ReX 1.4.0 opens its control port as unencrypted TCP but initializes the
    HTTP client with ``Tunnel.url``, which is Modal's HTTPS endpoint. Keep the
    upstream runtime and authentication protocol, but expose that control port
    through ``encrypted_ports`` so its URL and transport agree.
    """

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started; ignoring duplicate start().")
            return

        self.logger.info("Starting Modal sandbox")
        self._hooks.on_custom_step("Starting Modal sandbox")
        started = time.monotonic()
        token = self._get_token()
        encrypted, unencrypted, extra_kwargs = _tls_port_configuration(
            self._modal_kwargs, self._port
        )
        try:
            self._sandbox = await modal.Sandbox.create.aio(
                "/usr/bin/env",
                "bash",
                "-c",
                self._start_swerex_cmd(token),
                image=self._image,
                timeout=int(self._deployment_timeout),
                encrypted_ports=encrypted,
                unencrypted_ports=unencrypted,
                app=self._app,
                **extra_kwargs,
            )
            tunnels = await self._sandbox.tunnels.aio()
            tunnel = tunnels[self._port]
            elapsed = time.monotonic() - started
            log_url = await self.get_modal_log_url()
            self.logger.info("Sandbox %s created in %.2fs", self._sandbox.object_id, elapsed)
            self.logger.info("Modal logs: %s", log_url)
            await asyncio.sleep(1)
            self.logger.info("Starting SWE-ReX runtime at %s", tunnel.url)
            self._hooks.on_custom_step("Starting runtime")
            self._runtime = RemoteRuntime(
                host=tunnel.url,
                timeout=self._runtime_timeout,
                auth_token=token,
                logger=self.logger,
            )
            remaining = max(0, self._startup_timeout - elapsed)
            await self._wait_until_alive(timeout=remaining)
        except Exception as exc:
            details = [f"{type(exc).__name__}: {exc}"]
            if self._sandbox is not None:
                details.append(f"sandbox_id={self._sandbox.object_id}")
                try:
                    details.append(f"logs={await self.get_modal_log_url()}")
                except Exception:
                    pass
                try:
                    await self._sandbox.terminate.aio()
                except Exception:
                    pass
            self._runtime = None
            self._sandbox = None
            raise RuntimeError(
                "Modal sandbox started, but the SWE-ReX control runtime did not "
                "become reachable. " + " | ".join(details)
            ) from exc

SERVER_LOG = "/tmp/swerex-server.log"


class AssignmentDaytonaDeploymentConfig(DaytonaDeploymentConfig):
    """Same as upstream, but ``image`` also accepts a built ``daytona_sdk.Image``.

    Upstream types ``image`` as plain ``str``, matching a registry name, but
    Daytona's own API (``CreateSandboxFromImageParams``) also accepts a built
    ``daytona_sdk.Image`` for a from-Dockerfile build — the same shape
    `assignment.utils.image.build_testbed_image` returns for Modal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Any = "python:3.11"


class AssignmentDaytonaDeployment(DaytonaDeployment):
    """Daytona deployment whose server survives its own launch call.

    SWE-ReX 1.4.0 runs the (indefinitely-running) SWE-ReX HTTP server in the
    foreground of a session exec call, with ``runAsync=True``. That flag does
    not detach the process: Daytona's session exec API keeps the call open
    until the command stops producing output, then kills it and reports that
    kill as a failed, non-zero exit — even though the server logs show it
    came up cleanly (pipx installs, Uvicorn starts, then it gets shut back
    down right after). Upstream's ``response.exit_code != 0`` check then
    tears the sandbox down on every single launch.

    The fix is to background the server (``nohup ... & echo $!``, the same
    pattern `chess_sandbox.py` already uses for the chess server), so the
    exec call returns immediately with a real, successful exit code while the
    server keeps running detached from that call's stdout. Readiness is still
    confirmed for real by ``_wait_until_alive`` polling the HTTP endpoint,
    unchanged from upstream.

    Also fixes a second bug on the same path: upstream hands the whole
    ``PortPreviewUrl`` object to ``RemoteRuntime(host=...)`` instead of its
    ``.url``, which fails validation as soon as it's actually exercised.
    """

    def __init__(self, *, logger=None, **kwargs: Any) -> None:
        # Identical to `DaytonaDeployment.__init__`, except built against
        # `AssignmentDaytonaDeploymentConfig` so `image` can be a built
        # `daytona_sdk.Image`, not just a registry name.
        self._config = AssignmentDaytonaDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self._sandbox = None
        self._sandbox_id = None
        self.logger = logger or get_logger("rex-deploy")
        self._hooks = CombinedDeploymentHook()
        self._daytona = None
        self._auth_token = None

    def _get_command(self, *, token: str) -> str:
        foreground_command = super()._get_command(token=token)
        return f"nohup {foreground_command} > {SERVER_LOG} 2>&1 < /dev/null & echo $!"

    async def start(self) -> None:
        self._init_daytona()
        self.logger.info("Creating Daytona sandbox...")

        # A private sandbox's preview URL requires Daytona's own
        # `x-daytona-preview-token` header, which `RemoteRuntime` has no way
        # to send (it only ever sends its own `X-API-Key`). Making the
        # sandbox public removes that extra gate; access to the SWE-ReX
        # control server it exposes is still enforced by swerex's own
        # per-sandbox auth token, the same trust model already used for the
        # Modal backend's public HTTPS tunnel.
        params = CreateSandboxFromImageParams(image=self._config.image, public=True)
        assert self._daytona is not None

        self._sandbox = self._daytona.create(params)
        self._sandbox_id = self._sandbox.id
        self.logger.info("Created Daytona sandbox with ID: %s", self._sandbox_id)

        self._auth_token = self._get_token()
        command = self._get_command(token=self._auth_token)
        self.logger.info("Starting SWE Rex server in Daytona sandbox...")

        session_id = f"swerex-server-{uuid.uuid4().hex[:8]}"
        self._sandbox.process.create_session(session_id)

        request = SessionExecuteRequest(command=command, runAsync=True)
        response = self._sandbox.process.execute_session_command(session_id, request)
        if response.exit_code is not None and response.exit_code != 0:
            self.logger.error("Failed to start SWE Rex server: %s", response.output)
            await self.stop()
            raise RuntimeError(f"Failed to start SWE Rex server: {response.output}")

        host = self._sandbox.get_preview_link(self._config.port).url
        self._runtime = RemoteRuntime(host=host, port=None, auth_token=self._auth_token, logger=self.logger)

        started = time.time()
        try:
            await self._wait_until_alive(timeout=self._config.runtime_timeout)
        except Exception:
            # The backgrounded server may have since crashed; surface its log
            # instead of leaving only a bare polling timeout.
            try:
                logs = self._sandbox.process.exec(f"tail -n 80 {SERVER_LOG}", timeout=10)
                self.logger.error("SWE Rex server log:\n%s", logs.result)
            except Exception:
                pass
            raise
        self.logger.info("Runtime started in %.2fs", time.time() - started)


def _daytona_deployment(
    image: Any,
    deployment_timeout: float,
    startup_timeout: float,
    daytona_kwargs: dict[str, Any] | None,
) -> DaytonaDeployment:
    """Build a Daytona deployment, accepting a built image, not just a name."""
    kwargs = dict(daytona_kwargs or {})
    kwargs.setdefault("api_key", os.environ.get("DAYTONA_API_KEY", ""))
    kwargs.setdefault("container_timeout", deployment_timeout)
    kwargs.setdefault("runtime_timeout", startup_timeout)
    kwargs["image"] = image
    return AssignmentDaytonaDeployment(**kwargs)


class Environment:
    """
    Executes bash commands in a sandbox (Daytona by default, or Modal) via SWE-ReX.
    """

    # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py

    def __init__(
        self,
        image: "str | PurePath | modal.Image | Any" = "python:3.12",
        cwd: str = "/",
        startup_timeout: float = 600,
        runtime_timeout: float = 600,
        deployment_timeout: float = 600,
        install_pipx: bool = True,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        daytona_kwargs: dict[str, Any] | None = None,
        conda_env: str | None = None,
        backend: str | None = None,
    ):
        """Launch a sandbox and block until its runtime answers.

        Args:
            image: A prebuilt image (`modal.Image` or `daytona_sdk.Image`), a
                Dockerhub or ECR image name, or a path to a Dockerfile. Build a
                task testbed with `assignment.utils.image.build_testbed_image`,
                which is the only way to pass a credential to a private clone.
            cwd: Working directory for commands that do not specify one.
            startup_timeout: Seconds to wait for the SWE-ReX runtime to come up.
            runtime_timeout: Seconds a single command may run before timing out.
            deployment_timeout: Seconds the sandbox may stay alive before the
                provider reclaims it.
            install_pipx: Install pipx in the image, needed to bootstrap the
                SWE-ReX server when it is not already present. Modal backend only.
            modal_sandbox_kwargs: Additional keyword arguments forwarded to
                ``modal.Sandbox.create``. This is used for capabilities such as
                encrypted port forwarding. Ignored unless `backend` is "modal".
            daytona_kwargs: Additional keyword arguments forwarded to
                ``DaytonaDeployment``, such as ``target`` for the sandbox
                region. Ignored unless `backend` is "daytona".
            conda_env: Name of a conda environment to put on PATH for every
                command. SWE-bench images install the repository under test into
                an environment named ``testbed`` but never activate it, so
                without this ``python`` is conda's base environment, where the
                repository and its dependencies are not installed.
            backend: Sandbox provider, "daytona" or "modal". Defaults to
                `DEFAULT_BACKEND` (the `ASSIGNMENT_SANDBOX_BACKEND` environment
                variable, or "daytona").
        """
        self.cwd = cwd
        self.backend = (backend or DEFAULT_BACKEND).lower()
        # Merged into every command's environment; a per-call `env` wins.
        self.env_defaults: dict[str, str] = {}

        if self.backend == "daytona":
            self.deployment = _daytona_deployment(
                image=image,
                deployment_timeout=deployment_timeout,
                startup_timeout=startup_timeout,
                daytona_kwargs=daytona_kwargs,
            )
        elif self.backend == "modal":
            self.deployment = AssignmentModalDeployment(
                image=image,
                startup_timeout=startup_timeout,
                runtime_timeout=runtime_timeout,
                deployment_timeout=deployment_timeout,
                install_pipx=install_pipx,
                modal_sandbox_kwargs=modal_sandbox_kwargs,
            )
        else:
            raise ValueError(f"Unknown sandbox backend: {self.backend!r} (expected 'daytona' or 'modal')")

        async def _start():
            await self.deployment.start()
            await self.deployment.is_alive()

        asyncio.run(_start())

        if conda_env:
            self.activate_conda_env(conda_env)

        # Read the platform from inside the container, not from platform.uname(),
        # which would describe the machine running this code instead.
        self.system, self.release, self.version, self.machine = self.execute(
            "uname -s; uname -r; uname -v; uname -m"
        )["output"].splitlines()

    def is_alive(self) -> bool:
        """Whether the sandbox is still running.

        The provider reclaims a sandbox once `deployment_timeout` elapses, and
        the deployment keeps its handle afterwards, so this asks the sandbox
        itself rather than trusting the handle's existence.
        """
        if self.backend == "modal":
            sandbox = self.deployment._sandbox
            return sandbox is not None and sandbox.poll() is None

        try:
            asyncio.run(self.deployment.is_alive())
            return True
        except Exception:
            return False

    def activate_conda_env(self, name: str, root: str = "/opt/miniconda3") -> str:
        """Put a conda environment's bin directory first on PATH for all commands.

        Activating properly needs a login shell, which these commands do not
        get. Prepending the environment's bin directory has the same effect for
        ``python``, ``pip``, and anything else installed there.

        Args:
            name: The environment name, for example ``testbed``.
            root: Where conda is installed in the image.

        Returns:
            The PATH now used for every command.

        Raises:
            FileNotFoundError: If the environment is not in the image, rather
                than silently leaving the wrong interpreter on PATH.
        """
        binary_dir = posixpath.join(root, "envs", name, "bin")
        if self.execute(f"test -d {binary_dir}", cwd="/")["returncode"] != 0:
            raise FileNotFoundError(f"No conda environment at {binary_dir}")

        current = self.execute("printenv PATH", cwd="/")["output"].strip()
        self.env_defaults["PATH"] = f"{binary_dir}:{current}"
        return self.env_defaults["PATH"]

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        # Command defaults to shell=False, which requires `command` to be an argv
        # list; default to shell=True so a plain command string works.
        shell: bool | None = True,
        # check=False so a failing command returns its exit code instead of
        # raising and discarding the output the agent needs to see.
        check: bool = False,
    ) -> dict:
        """Run a command in the sandbox and return its result.

        Args:
            command: A shell string, or an argv list if `shell` is False.
            timeout: Seconds before the command is killed. None means no timeout.
            cwd: Working directory to run the command in. Defaults to the
                environment's `cwd`.
            env: Environment variables to set for the command.
            shell: Run the command through a shell, as with `subprocess.run`.
                None is treated as True, since a model calling this as a tool may
                send null for an omitted argument.
            check: Raise on a non-zero exit code instead of reporting it.

        Returns:
            A dict with `output`, `returncode`, and `exception_info`. `output`
            holds stdout and stderr interleaved, as they would appear in a
            terminal. `exception_info` is empty when the command ran, whatever
            its exit code. If the call itself fails (the sandbox died, the
            command timed out), the exception is caught rather than raised:
            `returncode` is -1, `exception_info` describes it, and an `extra` key
            carries the exception type.
        """
        # Command.shell is a strict bool, so None would fail validation.
        shell = True if shell is None else shell

        arguments = {
            "command": command,
            "timeout": timeout,
            "env": {**self.env_defaults, **(env or {})} or None,
            "shell": shell,
            "check": check,
            "merge_output_streams": False,
        }
        if cwd is not None:
            arguments["cwd"] = cwd
        # A cwd from the caller wins; otherwise the environment's default applies.
        arguments.setdefault("cwd", self.cwd)

        try:
            result = asyncio.run(self.deployment.runtime.execute(Command(**arguments)))
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                # The coding agent consumes one observation string, while the
                # task harness needs the streams separately for diagnostics.
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
            if result.stderr:
                output["output"] += result.stderr
        except Exception as e:
            # A command can fail for reasons worth reporting to the model, but a
            # sandbox that is gone is terminal: every later command fails the
            # same way, so raise rather than let the caller keep going.
            if not self.is_alive():
                raise RuntimeError(
                    "The sandbox is no longer running, so no further commands can "
                    f"be executed. Last error: {e}"
                ) from e

            # NOTE(source): https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/extra/swerex_modal.py#L82-L87
            # Same keys as the success path, so callers never have to branch on
            # which one they got.
            output = {
                "stdout": "",
                "stderr": str(e) if str(e) else "",
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return output

    def stop(self, timeout: float = 10):
        """Shut down the runtime and terminate the sandbox.

        Args:
            timeout: Seconds allowed for each of the shutdown and the
                termination steps.
        """

        async def _stop():
            if self.backend != "modal":
                # DaytonaDeployment.stop() already deletes the sandbox.
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
                return
            # ModalDeployment.stop() has an inverted poll() check and only
            # terminates sandboxes that have *already* exited, so a live sandbox
            # leaks until deployment_timeout. Terminate it explicitly. Grab the
            # reference first: stop() clears _sandbox.
            sandbox = self.deployment._sandbox
            try:
                await asyncio.wait_for(self.deployment.stop(), timeout=timeout)
            finally:
                if sandbox is not None:
                    await asyncio.wait_for(sandbox.terminate.aio(), timeout=timeout)

        asyncio.run(_stop())

    def tunnel_url(self, port: int) -> str:
        """Return the public URL for a port forwarded when the sandbox started."""

        if self.backend != "modal":
            # Daytona exposes a preview URL for any listening port on demand,
            # with no need to declare it when the sandbox was created.
            sandbox = self.deployment._sandbox
            if sandbox is None:
                raise RuntimeError("The sandbox is not running.")
            return sandbox.get_preview_link(port).url

        async def _tunnel_url() -> str:
            tunnels = await self.deployment.sandbox.tunnels.aio()
            try:
                return tunnels[port].url
            except KeyError as exc:
                forwarded = ", ".join(str(item) for item in sorted(tunnels)) or "none"
                raise ValueError(
                    f"Port {port} was not forwarded. Available forwarded ports: {forwarded}."
                ) from exc

        return asyncio.run(_tunnel_url())

    def __enter__(self):
        """Enter a `with` block. The sandbox is already running by this point."""
        return self

    def __exit__(self, *exc):
        """Terminate the sandbox on leaving a `with` block, including on error."""
        self.stop()
