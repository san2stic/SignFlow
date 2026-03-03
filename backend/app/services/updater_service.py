"""UpdaterService — Git polling + Docker build/deploy pipeline with WebSocket broadcast.

Architecture decision: Option A — module inside existing FastAPI backend.
The service runs a background asyncio task that polls git for new commits,
triggers docker compose up --build, streams logs, and handles rollback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import aiohttp
from fastapi import WebSocket

from app.models.deployment import DeploymentHistory

logger = logging.getLogger("signflow.updater")


class UpdaterService:
    """Singleton service that manages the Git→Docker update pipeline.

    Usage:
        service = UpdaterService(db_factory=SessionLocal)
        await service.start()   # at FastAPI startup
        await service.stop()    # at FastAPI shutdown
    """

    def __init__(
        self,
        db_factory: Callable,
        *,
        repo_path: str = "/repo",
        compose_file: str = "/repo/docker-compose.yml",
        poll_interval_s: int = 60,
        git_branch: str = "main",
        compose_service_name: str = "backend",
        health_check_url: str = "http://localhost:8000/healthz",
        health_check_retries: int = 5,
        health_check_delay_s: int = 3,
        enabled: bool = False,
    ) -> None:
        self._db_factory = db_factory
        self.repo_path = repo_path
        self.compose_file = compose_file
        self.poll_interval_s = poll_interval_s
        self.git_branch = git_branch
        self.compose_service_name = compose_service_name
        self.health_check_url = health_check_url
        self.health_check_retries = health_check_retries
        self.health_check_delay_s = health_check_delay_s
        self.enabled = enabled

        # Runtime state
        self._state: str = "idle"
        self._current_deployment_id: Optional[int] = None
        self._last_check_at: Optional[datetime] = None
        self._local_commit: Optional[str] = None
        self._remote_commit: Optional[str] = None
        self._consecutive_rollbacks: int = 0
        self._max_auto_rollbacks: int = 3

        # Async control
        self._poll_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # WebSocket broadcast pool
        self._ws_clients: set[WebSocket] = set()

    # ─────────────────────────── Lifecycle ────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if not self.enabled:
            logger.info("updater_service_disabled — skipping start")
            return
        if self._poll_task and not self._poll_task.done():
            logger.warning("updater_poll_loop_already_running")
            return
        self._poll_task = asyncio.create_task(self._poll_loop(), name="updater-poll")
        logger.info("updater_service_started", extra={"branch": self.git_branch, "interval_s": self.poll_interval_s})

    async def stop(self) -> None:
        """Stop the background polling loop gracefully."""
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("updater_service_stopped")

    # ─────────────────────────── Public API ───────────────────────────────────

    async def trigger_manual(self) -> int:
        """Trigger a manual deployment cycle immediately.

        Returns:
            deployment_id of the newly created deployment record.

        Raises:
            RuntimeError: if a deployment is already in progress.
        """
        if self._state not in ("idle", "success", "error", "rolled_back"):
            raise RuntimeError(f"A deployment is already in progress: state={self._state}")

        db = self._db_factory()
        try:
            entry = DeploymentHistory(
                status="fetching",
                triggered_by="manual",
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            deployment_id = entry.id
        finally:
            db.close()

        asyncio.create_task(
            self._run_pipeline(deployment_id),
            name=f"updater-pipeline-{deployment_id}",
        )
        return deployment_id

    async def get_status(self) -> dict:
        """Return the current status of the updater service."""
        last_deployment = None
        db = self._db_factory()
        try:
            row = (
                db.query(DeploymentHistory)
                .order_by(DeploymentHistory.created_at.desc())
                .first()
            )
            if row:
                last_deployment = {
                    "id": row.id,
                    "status": row.status,
                    "triggered_by": row.triggered_by,
                    "commit_hash": row.commit_hash,
                    "commit_message": row.commit_message,
                    "commit_author": row.commit_author,
                    "committed_at": row.committed_at.isoformat() if row.committed_at else None,
                    "build_duration_s": row.build_duration_s,
                    "deploy_duration_s": row.deploy_duration_s,
                    "total_duration_s": row.total_duration_s,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    "rollback_of_id": row.rollback_of_id,
                }
        finally:
            db.close()

        # Try to get git remote url
        git_remote_url = await self._get_git_remote_url()

        return {
            "state": self._state,
            "current_deployment_id": self._current_deployment_id,
            "last_deployment": last_deployment,
            "git_remote_url": git_remote_url,
            "git_branch": self.git_branch,
            "local_commit": self._local_commit,
            "remote_commit": self._remote_commit,
            "last_check_at": self._last_check_at.isoformat() if self._last_check_at else None,
            "poll_interval_s": self.poll_interval_s,
            "auto_update_enabled": self.enabled,
        }

    async def get_history(self, limit: int = 20) -> list[dict]:
        """Return the deployment history from the database."""
        db = self._db_factory()
        try:
            rows = (
                db.query(DeploymentHistory)
                .order_by(DeploymentHistory.created_at.desc())
                .limit(limit)
                .all()
            )
            result = []
            for row in rows:
                result.append({
                    "id": row.id,
                    "status": row.status,
                    "triggered_by": row.triggered_by,
                    "commit_hash": row.commit_hash,
                    "previous_commit_hash": row.previous_commit_hash,
                    "commit_message": row.commit_message,
                    "commit_author": row.commit_author,
                    "committed_at": row.committed_at.isoformat() if row.committed_at else None,
                    "build_log": row.build_log,
                    "error_message": row.error_message,
                    "build_duration_s": row.build_duration_s,
                    "deploy_duration_s": row.deploy_duration_s,
                    "total_duration_s": row.total_duration_s,
                    "rollback_of_id": row.rollback_of_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                })
            return result
        finally:
            db.close()

    async def rollback_to(self, deployment_id: int) -> int:
        """Trigger a manual rollback to a specific successful deployment.

        Args:
            deployment_id: ID of the target deployment to rollback to.

        Returns:
            ID of the new rollback deployment record.

        Raises:
            ValueError: if the target deployment doesn't exist or isn't successful.
            RuntimeError: if a deployment is already in progress.
        """
        if self._state not in ("idle", "success", "error", "rolled_back"):
            raise RuntimeError(f"A deployment is already in progress: state={self._state}")

        db = self._db_factory()
        try:
            target = db.query(DeploymentHistory).filter(DeploymentHistory.id == deployment_id).first()
            if not target:
                raise ValueError(f"Deployment {deployment_id} not found")
            if target.commit_hash is None:
                raise ValueError(f"Deployment {deployment_id} has no commit_hash — cannot rollback")

            target_commit = target.commit_hash

            # Create the rollback entry
            entry = DeploymentHistory(
                status="fetching",
                triggered_by="manual",
                rollback_of_id=deployment_id,
                previous_commit_hash=self._local_commit,
                commit_hash=target_commit,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            new_deployment_id = entry.id
        finally:
            db.close()

        asyncio.create_task(
            self._run_pipeline(new_deployment_id, target_commit=target_commit),
            name=f"updater-rollback-{new_deployment_id}",
        )
        return new_deployment_id

    # ─────────────────────────── WebSocket pool ───────────────────────────────

    async def register_ws(self, ws: WebSocket) -> None:
        """Register a WebSocket client for broadcast."""
        self._ws_clients.add(ws)
        logger.debug("ws_client_registered", extra={"total": len(self._ws_clients)})

    async def unregister_ws(self, ws: WebSocket) -> None:
        """Unregister a WebSocket client."""
        self._ws_clients.discard(ws)
        logger.debug("ws_client_unregistered", extra={"total": len(self._ws_clients)})

    async def _broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected WebSocket clients.

        Dead clients are removed silently.
        """
        if not self._ws_clients:
            return

        dead_clients: set[WebSocket] = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead_clients.add(ws)

        for ws in dead_clients:
            self._ws_clients.discard(ws)

    # ─────────────────────────── Poll loop ────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Infinite polling loop running as an asyncio background task."""
        logger.info("updater_poll_loop_start")
        while True:
            try:
                await asyncio.sleep(self.poll_interval_s)
                await self._poll_cycle()
            except asyncio.CancelledError:
                logger.info("updater_poll_loop_cancelled")
                break
            except Exception as exc:
                logger.error("updater_poll_loop_error", extra={"error": str(exc)})
                # Brief pause to avoid tight error loop
                await asyncio.sleep(10)

    async def _poll_cycle(self) -> None:
        """A single poll cycle: check for new commits and trigger build if needed."""
        async with self._lock:
            if self._state not in ("idle", "success", "error", "rolled_back"):
                # Already running — skip this tick
                return

        self._last_check_at = datetime.now(timezone.utc)

        try:
            local_head, remote_head = await self._git_fetch()
        except Exception as exc:
            logger.warning("updater_git_fetch_failed", extra={"error": str(exc)})
            return

        self._local_commit = local_head
        self._remote_commit = remote_head

        if local_head == remote_head:
            # No changes
            await self._broadcast({
                "type": "heartbeat",
                "ts": datetime.now(timezone.utc).isoformat(),
                "state": self._state,
                "local_commit": local_head,
                "remote_commit": remote_head,
            })
            return

        # New commit detected — create DB record and run pipeline
        logger.info(
            "updater_new_commit_detected",
            extra={"local": local_head, "remote": remote_head},
        )

        db = self._db_factory()
        try:
            entry = DeploymentHistory(
                status="fetching",
                triggered_by="auto",
                previous_commit_hash=local_head,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            deployment_id = entry.id
        finally:
            db.close()

        # Run pipeline in background (don't await — the poll loop should continue)
        asyncio.create_task(
            self._run_pipeline(deployment_id),
            name=f"updater-pipeline-{deployment_id}",
        )

    # ─────────────────────────── Pipeline orchestration ───────────────────────

    async def _run_pipeline(
        self, deployment_id: int, target_commit: str | None = None
    ) -> None:
        """Orchestrate all steps of the update pipeline.

        Steps:
            1. git pull (or git reset --hard if target_commit given)
            2. docker compose up --build --no-deps --wait
            3. health check
            4. mark success or rollback on failure
        """
        async with self._lock:
            self._state = "pulling"
            self._current_deployment_id = deployment_id

        started_at = time.monotonic()
        db = self._db_factory()

        try:
            # ── Step 1: Update state to 'pulling' ──────────────────────────
            entry = db.query(DeploymentHistory).filter(DeploymentHistory.id == deployment_id).first()
            if not entry:
                logger.error("updater_deployment_entry_missing", extra={"id": deployment_id})
                return

            entry.status = "pulling"
            entry.started_at = datetime.now(timezone.utc)
            db.commit()

            await self._broadcast({
                "type": "status_update",
                "ts": datetime.now(timezone.utc).isoformat(),
                "state": "pulling",
                "deployment_id": deployment_id,
            })

            # ── Step 2: Git pull or reset ──────────────────────────────────
            try:
                if target_commit:
                    await self._git_reset_to(target_commit)
                    new_commit = target_commit
                else:
                    new_commit = await self._git_pull()
            except Exception as exc:
                await self._fail_deployment(db, entry, str(exc), started_at)
                return

            entry.commit_hash = new_commit
            # Fetch commit metadata
            commit_msg, commit_author, committed_at = await self._get_commit_metadata(new_commit)
            entry.commit_message = commit_msg
            entry.commit_author = commit_author
            entry.committed_at = committed_at
            db.commit()
            self._local_commit = new_commit

            # ── Step 3: Docker build ───────────────────────────────────────
            async with self._lock:
                self._state = "building"

            entry.status = "building"
            db.commit()

            await self._broadcast({
                "type": "status_update",
                "ts": datetime.now(timezone.utc).isoformat(),
                "state": "building",
                "deployment_id": deployment_id,
                "commit_hash": new_commit,
            })

            build_start = time.monotonic()
            try:
                await self._docker_build_and_deploy(deployment_id, db, entry)
            except Exception as exc:
                # Rollback if we have a previous commit
                prev_hash = entry.previous_commit_hash
                if prev_hash:
                    logger.warning(
                        "updater_build_failed_triggering_rollback",
                        extra={"error": str(exc), "previous_hash": prev_hash},
                    )
                    await self._fail_deployment(db, entry, str(exc), started_at)
                    await self._rollback(deployment_id, prev_hash)
                else:
                    await self._fail_deployment(db, entry, str(exc), started_at)
                return

            build_duration = time.monotonic() - build_start
            entry.build_duration_s = build_duration
            db.commit()

            # ── Step 4: Health check ───────────────────────────────────────
            async with self._lock:
                self._state = "deploying"

            entry.status = "deploying"
            db.commit()

            await self._broadcast({
                "type": "status_update",
                "ts": datetime.now(timezone.utc).isoformat(),
                "state": "deploying",
                "deployment_id": deployment_id,
            })

            deploy_start = time.monotonic()
            health_ok = await self._health_check()
            deploy_duration = time.monotonic() - deploy_start

            if not health_ok:
                error_msg = "Health check failed after deployment"
                prev_hash = entry.previous_commit_hash
                if prev_hash:
                    await self._fail_deployment(db, entry, error_msg, started_at)
                    await self._rollback(deployment_id, prev_hash)
                else:
                    await self._fail_deployment(db, entry, error_msg, started_at)
                return

            # ── Step 5: Mark success ───────────────────────────────────────
            total_duration = time.monotonic() - started_at

            async with self._lock:
                self._state = "success"
                self._current_deployment_id = None

            entry.status = "success"
            entry.deploy_duration_s = deploy_duration
            entry.total_duration_s = total_duration
            entry.completed_at = datetime.now(timezone.utc)
            db.commit()

            self._consecutive_rollbacks = 0

            await self._broadcast({
                "type": "completed",
                "ts": datetime.now(timezone.utc).isoformat(),
                "deployment_id": deployment_id,
                "commit_hash": new_commit,
                "duration_s": total_duration,
            })

            logger.info(
                "updater_deployment_success",
                extra={"deployment_id": deployment_id, "commit": new_commit, "duration_s": total_duration},
            )

        except Exception as exc:
            logger.error("updater_pipeline_unexpected_error", extra={"error": str(exc)}, exc_info=True)
            try:
                db_inner = self._db_factory()
                try:
                    row = db_inner.query(DeploymentHistory).filter(
                        DeploymentHistory.id == deployment_id
                    ).first()
                    if row and row.status not in ("success", "rolled_back"):
                        row.status = "error"
                        row.error_message = str(exc)
                        row.completed_at = datetime.now(timezone.utc)
                        row.total_duration_s = time.monotonic() - started_at
                        db_inner.commit()
                finally:
                    db_inner.close()
            except Exception:
                pass

            async with self._lock:
                self._state = "error"
                self._current_deployment_id = None

            await self._broadcast({
                "type": "error",
                "ts": datetime.now(timezone.utc).isoformat(),
                "deployment_id": deployment_id,
                "message": str(exc),
            })
        finally:
            db.close()

    async def _fail_deployment(
        self,
        db,
        entry: DeploymentHistory,
        error_msg: str,
        started_at: float,
    ) -> None:
        """Mark a deployment as failed, update DB and broadcast."""
        async with self._lock:
            self._state = "error"
            self._current_deployment_id = None

        entry.status = "error"
        entry.error_message = error_msg
        entry.completed_at = datetime.now(timezone.utc)
        entry.total_duration_s = time.monotonic() - started_at
        db.commit()

        await self._broadcast({
            "type": "error",
            "ts": datetime.now(timezone.utc).isoformat(),
            "deployment_id": entry.id,
            "message": error_msg,
        })
        logger.error("updater_deployment_failed", extra={"deployment_id": entry.id, "error": error_msg})

    # ─────────────────────────── Git helpers ──────────────────────────────────

    async def _git_fetch(self) -> tuple[str, str]:
        """Fetch remote and compare local HEAD vs remote HEAD.

        Returns:
            (local_head, remote_head) as full commit hashes.
        """
        # 1. git fetch --quiet
        fetch_proc = await asyncio.create_subprocess_exec(
            "git", "fetch", "--quiet",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, fetch_err = await fetch_proc.communicate()
        if fetch_proc.returncode != 0:
            raise RuntimeError(f"git fetch failed: {fetch_err.decode().strip()}")

        # 2. git rev-parse HEAD (local)
        local_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        local_out, _ = await local_proc.communicate()
        local_head = local_out.decode().strip()

        # 3. git rev-parse origin/<branch> (remote tracking)
        remote_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", f"origin/{self.git_branch}",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        remote_out, remote_err = await remote_proc.communicate()
        if remote_proc.returncode != 0:
            raise RuntimeError(f"git rev-parse origin/{self.git_branch} failed: {remote_err.decode().strip()}")
        remote_head = remote_out.decode().strip()

        return local_head, remote_head

    async def _git_pull(self) -> str:
        """Pull latest commits on the configured branch.

        Returns:
            New local HEAD commit hash after pull.
        """
        pull_proc = await asyncio.create_subprocess_exec(
            "git", "pull", "--ff-only", "origin", self.git_branch,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, pull_err = await pull_proc.communicate()
        if pull_proc.returncode != 0:
            raise RuntimeError(f"git pull failed: {pull_err.decode().strip()}")

        # Get new HEAD
        head_proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
        )
        head_out, _ = await head_proc.communicate()
        return head_out.decode().strip()

    async def _git_reset_to(self, commit_hash: str) -> None:
        """Hard-reset to a specific commit hash (used for rollback)."""
        reset_proc = await asyncio.create_subprocess_exec(
            "git", "reset", "--hard", commit_hash,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, reset_err = await reset_proc.communicate()
        if reset_proc.returncode != 0:
            raise RuntimeError(f"git reset --hard {commit_hash} failed: {reset_err.decode().strip()}")

    async def _get_commit_metadata(
        self, commit_hash: str
    ) -> tuple[Optional[str], Optional[str], Optional[datetime]]:
        """Retrieve commit message, author and timestamp for given hash."""
        try:
            # Format: message|author|ISO timestamp
            log_proc = await asyncio.create_subprocess_exec(
                "git", "log", "-1", "--pretty=format:%s|%an <%ae>|%aI", commit_hash,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await log_proc.communicate()
            if log_proc.returncode != 0:
                return None, None, None
            parts = out.decode().strip().split("|", 2)
            msg = parts[0] if len(parts) > 0 else None
            author = parts[1] if len(parts) > 1 else None
            ts_str = parts[2] if len(parts) > 2 else None
            committed_at: Optional[datetime] = None
            if ts_str:
                try:
                    committed_at = datetime.fromisoformat(ts_str)
                except ValueError:
                    pass
            return msg, author, committed_at
        except Exception:
            return None, None, None

    async def _get_git_remote_url(self) -> str:
        """Retrieve origin remote URL from git config."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "remote", "get-url", "origin",
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                return out.decode().strip()
        except Exception:
            pass
        return ""

    # ─────────────────────────── Docker helpers ───────────────────────────────

    async def _docker_build_and_deploy(
        self, deployment_id: int, db, entry: DeploymentHistory
    ) -> None:
        """Run docker compose up --build and stream logs.

        Uses Strategy A from the architecture doc:
            docker compose -f <compose_file> up --build --no-deps --wait <service>

        Raises:
            RuntimeError: if docker compose exits with non-zero code.
        """
        compose_dir = self._get_compose_dir()

        proc = await asyncio.create_subprocess_exec(
            "docker", "compose",
            "-f", self.compose_file,
            "up", "--build", "--no-deps", "--wait",
            self.compose_service_name,
            cwd=compose_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout for unified log
        )

        # Stream logs line by line and accumulate in DB
        log_buffer: list[str] = []
        await self._stream_logs(proc, deployment_id, log_buffer)

        # Wait for process completion
        return_code = await proc.wait()

        # Persist accumulated log
        full_log = "\n".join(log_buffer)
        entry.build_log = full_log
        db.commit()

        if return_code != 0:
            raise RuntimeError(
                f"docker compose up --build exited with code {return_code}. "
                f"Last 500 chars of log:\n{full_log[-500:]}"
            )

    async def _stream_logs(
        self,
        proc: asyncio.subprocess.Process,
        deployment_id: int,
        log_buffer: list[str],
    ) -> None:
        """Read stdout from a subprocess and broadcast each line via WS."""
        if proc.stdout is None:
            return

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            log_buffer.append(line)

            await self._broadcast({
                "type": "build_log",
                "ts": datetime.now(timezone.utc).isoformat(),
                "deployment_id": deployment_id,
                "line": line,
            })

    def _get_compose_dir(self) -> str:
        """Return directory containing the compose file."""
        import os
        return os.path.dirname(self.compose_file) or self.repo_path

    # ─────────────────────────── Health check ─────────────────────────────────

    async def _health_check(self) -> bool:
        """Perform repeated GET requests on /healthz until 200 or timeout.

        Returns:
            True if service is healthy, False if all retries exhausted.
        """
        for attempt in range(self.health_check_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.health_check_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(
                                "updater_health_check_ok",
                                extra={"attempt": attempt + 1},
                            )
                            return True
            except Exception as exc:
                logger.debug(
                    "updater_health_check_attempt_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )

            # Wait before retrying
            await asyncio.sleep(self.health_check_delay_s)

        logger.warning(
            "updater_health_check_exhausted",
            extra={"retries": self.health_check_retries},
        )
        return False

    # ─────────────────────────── Rollback ─────────────────────────────────────

    async def _rollback(self, failed_deployment_id: int, previous_hash: str) -> None:
        """Automatically rollback to a previous commit after a build failure.

        Creates a new DB entry for the rollback deployment and runs the pipeline
        with the target commit hash.

        Safety: if more than _max_auto_rollbacks consecutive rollbacks occur,
        the service halts further attempts.
        """
        self._consecutive_rollbacks += 1
        if self._consecutive_rollbacks > self._max_auto_rollbacks:
            logger.error(
                "updater_max_rollbacks_exceeded",
                extra={"consecutive": self._consecutive_rollbacks, "limit": self._max_auto_rollbacks},
            )
            await self._broadcast({
                "type": "error",
                "ts": datetime.now(timezone.utc).isoformat(),
                "deployment_id": failed_deployment_id,
                "message": (
                    f"Auto-rollback blocked: {self._consecutive_rollbacks} consecutive rollbacks "
                    f"exceeded limit of {self._max_auto_rollbacks}. Manual intervention required."
                ),
            })
            return

        db = self._db_factory()
        try:
            # Retrieve current commit to set as previous for the rollback entry
            current_local = self._local_commit

            rollback_entry = DeploymentHistory(
                status="fetching",
                triggered_by="auto",
                rollback_of_id=failed_deployment_id,
                previous_commit_hash=current_local,
                commit_hash=previous_hash,
            )
            db.add(rollback_entry)
            db.commit()
            db.refresh(rollback_entry)
            rollback_id = rollback_entry.id
        finally:
            db.close()

        await self._broadcast({
            "type": "rollback",
            "ts": datetime.now(timezone.utc).isoformat(),
            "deployment_id": rollback_id,
            "rollback_of_id": failed_deployment_id,
            "state": "deploying",
            "commit_hash": previous_hash,
        })

        logger.info(
            "updater_auto_rollback_triggered",
            extra={
                "rollback_deployment_id": rollback_id,
                "target_commit": previous_hash,
                "failed_deployment_id": failed_deployment_id,
            },
        )

        asyncio.create_task(
            self._run_pipeline(rollback_id, target_commit=previous_hash),
            name=f"updater-rollback-{rollback_id}",
        )


# ──────────────────────── Module-level singleton ──────────────────────────────
# Instantiated lazily at app startup via main.py using actual config values.
# The reference is set by backend/app/main.py after reading settings.
updater_service: Optional[UpdaterService] = None
