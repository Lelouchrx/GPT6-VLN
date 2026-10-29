
import asyncio
import base64
import json
import os
import sys

EVENT = {"type": "object", "additionalProperties": False, "properties": {
    "clause": {"type": "string"}, "state": {"type": "string", "enum": ["pending", "seen", "reached", "passed"]},
    "frames": {"type": "array", "maxItems": 3, "items": {"type": "integer"}},
    "observation": {"type": "string"}}, "required": ["clause", "state", "frames", "observation"]}
NAV = {"type": "object", "additionalProperties": False, "properties": {
    "skill": {"type": "string", "enum": ["move", "stop", "backtrack"]},
    "turn_deg": {"type": "integer", "enum": list(range(-180,181,15))},
    "distance_cm": {"type": "integer", "enum": list(range(0,251,25))},
    "node_id": {"type": "integer"}, "stage": {"type": "integer"},
    "place": {"type": "string"}, "note": {"type": "string"},
    "events": {"type": "array", "maxItems": 8, "items": EVENT}},
    "required": ["skill", "turn_deg", "distance_cm", "node_id", "stage", "place", "note", "events"]}
PROMPT = ''
