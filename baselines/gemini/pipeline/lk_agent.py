#!/usr/bin/env python3
"""
LiveKit Voice Agent for Google Gemini Live realtime models.

Supported providers:
  gemini-2.5  – Google Gemini 2.5 Live API (plugin default model)
  gemini-3.1  – Google Gemini 3.1 Flash Live (preview)

The provider is resolved from the room-name prefix (e.g. room
'gemini-3.1-batch-0-t1' -> gemini-3.1), falling back to LK_PROVIDER.

Usage:
    # Development mode (connects to LiveKit Cloud, auto-dispatches on room join):
    python lk_agent.py dev

    # Console mode (runs locally in terminal, uses mic/speaker):
    python lk_agent.py console

Requirements:
    pip install "livekit-agents[google]~=1.3" python-dotenv

Environment variables (in .env.local):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    GOOGLE_API_KEY                (for Google Gemini)
    GOOGLE_VOICE                  (voice name, default: Puck)
    LK_PROVIDER                   (default provider: gemini-2.5 | gemini-3.1)
    LK_SYSTEM_PROMPT_FILE         (path to system prompt .txt, optional)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env.local")
log = logging.getLogger("lk_agent")
SYSTEM_PROMPT_FILE = Path(
    os.environ.get("LK_SYSTEM_PROMPT_FILE", str(SCRIPT_DIR / "system_prompt.txt"))
).expanduser()
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text().strip()
log.info("System prompt loaded from %s", SYSTEM_PROMPT_FILE.resolve())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROVIDER = os.getenv("LK_PROVIDER", "gemini-2.5")

# Order matters for prefix matching: longer/more specific names first.
KNOWN_PROVIDERS = ["gemini-3.1", "gemini-2.5"]


def provider_from_room(room_name: str) -> str | None:
    """Extract provider from room name prefix (e.g. 'gemini-2.5-batch-0' -> 'gemini-2.5')."""
    for p in KNOWN_PROVIDERS:
        if room_name.startswith(p):
            return p
    return None


def get_realtime_model(provider: str):
    """Return a Gemini RealtimeModel instance for the given provider."""
    from livekit.plugins import google

    voice = os.getenv("GOOGLE_VOICE", "Puck")

    if provider == "gemini-2.5":
        return google.realtime.RealtimeModel(voice=voice)

    if provider == "gemini-3.1":
        return google.realtime.RealtimeModel(
            model="gemini-3.1-flash-live-preview",
            voice=voice,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Set LK_PROVIDER to one of: {', '.join(KNOWN_PROVIDERS)}"
    )


# ---------------------------------------------------------------------------
# Agent definition & server
# ---------------------------------------------------------------------------
class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    room_name = ctx.room.name or ""
    provider = provider_from_room(room_name) or PROVIDER
    log.info("Room '%s' → provider '%s'", room_name, provider)

    model = get_realtime_model(provider)
    session = AgentSession(llm=model)

    await session.start(
        room=ctx.room,
        agent=VoiceAgent(),
    )
    # No greeting: the agent stays silent until the client streams the
    # speaker's audio, then responds full-duplex to that conversation.


if __name__ == "__main__":
    agents.cli.run_app(server)
