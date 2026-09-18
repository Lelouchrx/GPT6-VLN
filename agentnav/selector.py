"""VLN policy client: history frames + annotated current view -> action chunk.

The prompt is vln_real/policy.py's PROMPT_TEMPLATE verbatim, with the output
contract appended. History frames are plain RGB (no markers): a numbered circle
means a different 3D point in every frame, so annotating history invites the
model to conflate ids across time.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from .actions import ActionId, SubAction, describe, parse_chunk, to_primitives
from .config import SelectorConfig
from .types import Waypoint

SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)


@dataclass
class Decision:
    """The parsed action chunk, plus what the model actually said."""

    chunk: List[SubAction] = field(default_factory=list)
    primitives: List[int] = field(default_factory=list)
    raw: str = ""

    @property
    def stops_at_end(self) -> bool:
        """True when the chunk ends in STOP, i.e. move first, then finish."""
        return bool(self.chunk) and self.chunk[-1].action == ActionId.STOP

    @property
    def is_stop_only(self) -> bool:
        """True when STOP is the whole answer and there is nothing to execute."""
        return len(self.chunk) == 1 and self.chunk[0].action == ActionId.STOP

    @property
    def text(self) -> str:
        return describe(self.chunk)


@dataclass(frozen=True)
class CodexCredentials:
    base_url: str
    api_key: str
    model: str


def load_codex_credentials(config_path: Path, auth_path: Path) -> CodexCredentials:
    """Read provider URL, API key and model name from the Codex config."""
    text = config_path.read_text(encoding="utf-8")
    provider_match = re.search(r'^model_provider\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not provider_match:
        raise RuntimeError("model_provider is missing from Codex config")
    provider = re.escape(provider_match.group(1))
    section = re.search(rf'^\[model_providers\.{provider}\]\s*$([\s\S]*?)(?=^\[|\Z)', text, re.MULTILINE)
    if not section:
        raise RuntimeError("selected provider section is missing from Codex config")
    base_match = re.search(r'^base_url\s*=\s*"([^"]+)"', section.group(1), re.MULTILINE)
    if not base_match:
        raise RuntimeError("base_url is missing from selected Codex provider")
    model_match = re.search(r'^model\s*=\s*"([^"]+)"', text, re.MULTILINE)
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    api_key = auth.get("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from Codex auth file")
    return CodexCredentials(
        base_match.group(1).rstrip("/"),
        api_key,
        model_match.group(1) if model_match else "gpt-5.6-sol",
    )


def build_prompt(instruction: str, waypoints: Sequence[Waypoint], config: SelectorConfig,
                 history_count: int, text_history: Optional[Any] = None) -> str:
    lines = [PROMPT_TEMPLATE.format(instruction)]
    if history_count:
        lines.append(
            f"The first {history_count} images are your past observations, oldest first, "
            "uniformly sampled over the whole episode. The last image is your current view."
        )
    else:
        lines.append("The single image is your current view.")
    if waypoints:
        compact = [
            {
                "id": item.waypoint_id,
                "bearing_deg": round(item.bearing_deg, 1),
                "distance_m": round(item.distance_m, 2),
            }
            for item in waypoints
        ]
        lines.append(
            "Numbered green circles drawn on the current view mark floor positions the robot "
            "can reach; orange circles mark unexplored directions. They are reference only -- "
            "answer with motion, not with an id. Geometry: " + json.dumps(compact, ensure_ascii=True)
        )
    lo_cm, hi_cm = config.forward_cm_range
    lo_deg, hi_deg = config.turn_deg_range
    lines.append(
        "Answer with one or more comma-separated motion commands, in execution order:\n"
        f"  forward <{lo_cm:.0f}-{hi_cm:.0f}>      move ahead that many centimetres\n"
        f"  turn left <{lo_deg:.0f}-{hi_deg:.0f}>   rotate left that many degrees\n"
        f"  turn right <{lo_deg:.0f}-{hi_deg:.0f}>  rotate right that many degrees\n"
        "  stop                    the task is complete\n"
        "Turns happen in place. Use as many commands as the situation warrants, but remember "
        "you cannot see around a corner until you have turned. "
        "Wrap the final answer in <answer></answer> tags and write nothing after it.\n"
        "Example: <answer>turn left 30, forward 75</answer>"
    )
    if text_history:
        lines.append(
            "Your previous commands and what actually happened:\n"
            + json.dumps(text_history, ensure_ascii=True, indent=None)
            + "\nIf a command came back BLOCKED the way ahead is closed: turn to face a "
            "different direction instead of repeating it."
        )
    return "\n\n".join(lines)


class VLNSelector:
    """Single selector: an OpenAI-compatible Responses endpoint returning free text."""

    def __init__(
        self,
        config_path: Path,
        auth_path: Path,
        model: Optional[str] = None,
        timeout_s: float = 180.0,
        config: Optional[SelectorConfig] = None,
    ):
        self.credentials = load_codex_credentials(config_path, auth_path)
        self.model = model or self.credentials.model
        self.timeout_s = timeout_s
        self.config = config or SelectorConfig()
        self.calls = 0

    def select(
        self,
        instruction: str,
        history_rgbs: Sequence[Any],
        annotated_rgb,
        waypoints: Sequence[Waypoint],
        text_history: Optional[Any] = None,
    ) -> Decision:
        history_rgbs = list(history_rgbs or [])
        content: List[Dict[str, Any]] = [{
            "type": "input_text",
            "text": build_prompt(instruction, waypoints, self.config, len(history_rgbs), text_history),
        }]
        for frame in history_rgbs:
            content.append({"type": "input_image", "image_url": self._image_data_url(frame)})
        content.append({"type": "input_image", "image_url": self._image_data_url(annotated_rgb)})

        body = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.config.max_output_tokens,
        }
        text = _response_text(self._post(body))
        chunk = parse_chunk(text, self.config.forward_cm_range, self.config.turn_deg_range)
        self.calls += 1
        return Decision(
            chunk=chunk,
            primitives=to_primitives(chunk, self.config.forward_step_m, self.config.turn_angle_deg),
            raw=text.strip(),
        )

    def _post(self, body: Dict) -> Dict:
        request = urllib.request.Request(
            self.credentials.base_url + "/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.credentials.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "OpenAI/Python AgentNav",
                "X-Stainless-Lang": "python",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise RuntimeError(f"Responses API HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _image_data_url(rgb) -> str:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("failed to encode RGB observation")
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _response_text(response: Dict) -> str:
    if isinstance(response.get("output_text"), str) and response["output_text"].strip():
        return response["output_text"]
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise RuntimeError(f"Responses API result has no output_text: {json.dumps(response)[:400]}")
