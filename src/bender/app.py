"""Main application — wires FastAPI, slack-bolt, and all modules together."""

import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from bender import __version__
from bender.api import create_api
from bender.config import Settings
from bender.process_pool import ProcessPool
from bender.session_manager import SessionManager
from bender.slack_handler import register_handlers

logger = logging.getLogger(__name__)


class BenderApp:
    """Container for all application components."""

    def __init__(
        self,
        fastapi_app: FastAPI,
        bolt_app: AsyncApp,
        socket_handler: AsyncSocketModeHandler,
        settings: Settings,
        pool: ProcessPool,
    ) -> None:
        self.fastapi_app = fastapi_app
        self.bolt_app = bolt_app
        self.socket_handler = socket_handler
        self.settings = settings
        self.pool = pool


def create_app(settings: Settings) -> BenderApp:
    """Create and configure the Bender application."""
    sessions = SessionManager(db_path=settings.bender_db_path)
    pool = ProcessPool(
        workspace=settings.bender_workspace,
        sessions=sessions,
        backend=settings.bender_backend,
    )

    # Slack bolt app (Socket Mode)
    bolt_app = AsyncApp(token=settings.slack_bot_token)
    register_handlers(bolt_app, sessions, pool)
    socket_handler = AsyncSocketModeHandler(bolt_app, settings.slack_app_token)

    # FastAPI app
    fastapi_app = FastAPI(title="Bender API", version=__version__)
    create_api(fastapi_app, bolt_app.client, settings, sessions)

    return BenderApp(
        fastapi_app=fastapi_app,
        bolt_app=bolt_app,
        socket_handler=socket_handler,
        settings=settings,
        pool=pool,
    )


async def start(app: BenderApp, settings: Settings) -> None:
    """Start both Slack Socket Mode and FastAPI server concurrently."""
    logger.info("Starting Slack Socket Mode handler")
    logger.info("Starting FastAPI server on port %d", settings.bender_api_port)

    app.pool.start_reaper()

    uvicorn_config = uvicorn.Config(
        app.fastapi_app,
        host="0.0.0.0",
        port=settings.bender_api_port,
        log_level=settings.log_level.lower(),
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    try:
        results = await asyncio.gather(
            app.socket_handler.start_async(),
            uvicorn_server.serve(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error("Component failed: %s", result)
    finally:
        await app.pool.stop()
