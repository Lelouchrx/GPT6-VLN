"""Agent-SDK side of the skill loop: one session per episode, ten tools.

Runs under the claude-agent-sdk conda env (py3.10+); the Habitat env it drives
lives in the streamvln (py3.9) process on the other end of the Unix socket in
`socket_path`. Every tool here is a thin client -- all navigation logic and all
episode state belong to gpt_vln/skill_server.py.

Reads one JSON request on stdin, writes one JSON summary on stdout, exactly
like gpt_vln/sdk_bridge.py.
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import Annotated

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
    TextBlock, create_sdk_mcp_server, tool,
)

from gpt_vln.skill_modes import mode_config

SOCKET_PATH = None
CALLS = []
RPC_TIMEOUT_S = 180
MAX_RPC_RESPONSE_BYTES = 16 * 1024 * 1024


def rpc(method, **params):
    """One connect per call: no framing state to desynchronise."""
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(RPC_TIMEOUT_S)
    try:
        connection.connect(SOCKET_PATH)
        connection.sendall((json.dumps({"method": method, "params": params}) + "\n").encode())
        chunks = []
        size = 0
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RPC_RESPONSE_BYTES:
                raise RuntimeError("skill response exceeded 16 MiB")
            if b"\n" in chunk:
                break
    finally:
        connection.close()
    result = json.loads(b"".join(chunks).decode())
    CALLS.append({"skill": method, "params": params, "result": result})
    return result


def as_text(result):
    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            **({"is_error": True} if isinstance(result, dict) and result.get("error") else {})}


@tool("observe", "Look at the robot's current forward-facing camera. Returns an image "
                 "path to Read plus the remaining movement budget.", {})
async def observe(_args):
    return as_text(rpc("observe"))


@tool("observe_panorama",
      "Look around: returns three image paths (LEFT, FRONT, RIGHT) covering 360 degrees. "
      "Read them with the Read tool. Pixel coordinates passed to navigate_to must come "
      "from one of these views.", {})
async def observe_panorama(_args):
    return as_text(rpc("observe_panorama"))


@tool("observe_local_map",
      "Inspect current traversable geometry from recent RGB-D. Returns an agent-centered "
      "local occupancy image (facing up, light=free, dark=occupied, gray=unknown, 1m grid) "
      "plus optional walkable candidate waypoints.", {})
async def observe_local_map(_args):
    return as_text(rpc("observe_local_map"))


@tool("navigate_to",
      "Walk to a point you can see. Give the view name and the pixel coordinates within "
      "that view's image. Pick a point on visible traversable floor -- never through a "
      "wall or an inferred/hidden location. Optionally append short adjustment actions to "
      "run on arrival. Uses one of your limited movement budget.",
      {"view": Annotated[str, "LEFT, FRONT or RIGHT"],
       "u": Annotated[float, "horizontal pixel in that view"],
       "v": Annotated[float, "vertical pixel in that view"],
       "then_actions": Annotated[str, "optional, e.g. 'TURN_RIGHT(30)'"]})
async def navigate_to(args):
    return as_text(rpc("navigate_to", view=args.get("view"), u=args.get("u"),
                        v=args.get("v"), then_actions=args.get("then_actions", "")))


@tool("move",
      "Make a small local adjustment: up to 3 of FORWARD(25-75 cm), TURN_LEFT(15-45 deg), "
      "TURN_RIGHT(15-45 deg), with explicit magnitudes, e.g. 'TURN_LEFT(30) FORWARD(50)'. "
      "Uses one of your limited movement budget.",
      {"actions": Annotated[str, "the action string"]})
async def move(args):
    return as_text(rpc("move", actions=args.get("actions", "")))


@tool("go_back",
      "Recover from a wrong move: walk back to an earlier checkpoint and restore its "
      "heading. Every navigate_to and move records a checkpoint first; recall() lists "
      "them with their ids. The robot really walks/turns back, so this uses one of "
      "your limited movement budget.",
      {"checkpoint_id": Annotated[int, "id from recall()"]})
async def go_back(args):
    return as_text(rpc("go_back", checkpoint_id=args.get("checkpoint_id")))


@tool("recall",
      "Replay recorded state: remaining budget, recent movement outcomes, failed targets, "
      "checkpoints you can go_back to, and any subtasks or progress notes you chose to keep.", {})
async def recall(_args):
    return as_text(rpc("recall"))


@tool("note_progress",
      "Optional: record which subtask you are now on and the visual evidence for it. "
      "Evidence is required and must be something you actually see. Progress never moves backwards.",
      {"stage": Annotated[int, "1-based subtask number"],
       "evidence": Annotated[str, "what you see that justifies this"]})
async def note_progress(args):
    return as_text(rpc("note_progress", stage=args.get("stage"),
                        evidence=args.get("evidence", "")))


@tool("refine_instruction",
      "Optional: break the instruction into an ordered list of concrete subtasks, or "
      "correct an earlier reading. Use it if it helps; skip it if you can navigate without it.",
      {"subtasks": Annotated[list, "ordered list of short subtask strings"],
       "reason": Annotated[str, "why this reading, or why you are correcting it"]})
async def refine_instruction(args):
    return as_text(rpc("refine_instruction", subtasks=args.get("subtasks"),
                        reason=args.get("reason", "")))


@tool("stop",
      "End the episode. Cannot be undone. Evidence is required: say what you currently "
      "see that justifies stopping (arrived, or budget gone and you did not arrive).",
      {"evidence": Annotated[str, "what you see, or why you are stopping short"]})
async def stop(args):
    return as_text(rpc("stop", evidence=args.get("evidence", "")))


TOOLS = [observe, observe_panorama, observe_local_map, navigate_to, move, go_back,
         recall, note_progress, refine_instruction, stop]


async def run(request):
    global SOCKET_PATH, CALLS
    SOCKET_PATH = request["socket_path"]
    CALLS = []

    config = mode_config(request.get("skill_mode"))
    tools = [item for item in TOOLS if item.name in config["tools"]]
    server = create_sdk_mcp_server("vln", tools=tools)
    options = ClaudeAgentOptions(
        mcp_servers={"vln": server},
        allowed_tools=["Read"] + [f"mcp__vln__{t.name}" for t in tools],
        system_prompt=config["system_prompt"],
        permission_mode="dontAsk",
        model=request.get("model") or None,
        cli_path=request.get("claude_command") or "claude",
        max_turns=request.get("max_agent_turns", 80),
        # A Read of an image inlines it as base64 on one stream-json line; a panorama
        # blows past the 1MB default (CLIJSONDecodeError). Same fix as sdk_bridge.py.
        max_buffer_size=64 * 1024 * 1024,
    )

    budget = ", ".join(config["budget_skills"])
    prompt = (
        f"Navigation instruction: {request['instruction']}\n\n"
        f"You may use at most {request['max_moves']} movement skills "
        f"({budget}). Looking and remembering are free.\n\n"
        + (str(request["extra_prompt"]).strip() + "\n\n" if request.get("extra_prompt") else "")
        + "Begin."
    )

    final_text, usage = "", {}
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text = block.text
            elif isinstance(message, ResultMessage):
                usage = {"total_cost_usd": message.total_cost_usd,
                         "num_turns": message.num_turns,
                         "duration_ms": message.duration_ms,
                         "stop_reason": message.stop_reason}
    return {"final_text": final_text, "skill_calls": CALLS, "usage": usage}


def main():
    request = json.loads(sys.stdin.read())
    try:
        result = asyncio.run(run(request))
    except Exception as exc:  # report the partial transcript rather than losing it
        result = {"final_text": "", "skill_calls": CALLS, "bridge_error": repr(exc)}
        json.dump(result, sys.stdout)
        sys.stdout.flush()
        raise
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
