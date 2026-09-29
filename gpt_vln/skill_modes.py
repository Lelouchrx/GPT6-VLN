"""Three-column skill-interface ablation.

The claim is not a better navigation recipe. It is: after the workflow is taken
out of the code, does the model use the extra skills, and do they help?

  minimal   observe + move + stop. MIP-style interface; extra skills absent.
  optional  all skills visible; no prescribed order (current default).
  forced    all skills visible; prompt requires refine / note / recall+go_back.

Keep Habitat, budgets, and skill implementations identical across columns.
Only the tool list and the system prompt change.
"""
from __future__ import annotations

from collections import Counter

MODES = ("minimal", "optional", "forced")
ALL_SKILLS = (
    "observe", "observe_panorama", "observe_local_map",
    "navigate_to", "move", "go_back",
    "recall", "note_progress", "refine_instruction", "stop",
)
MINIMAL_SKILLS = ("observe", "move", "stop")
OPTIONAL_SKILLS = ALL_SKILLS
FORCED_SKILLS = ALL_SKILLS
# Skills whose use answers "did the model reach for the extra interface?"
EXTRA_SKILLS = tuple(s for s in ALL_SKILLS if s not in MINIMAL_SKILLS)

PROMPT_MINIMAL = (
    "You are a robot carrying out an indoor navigation instruction. You act only through "
    "the skills listed below. Nothing is shown automatically. Choose whichever skills you "
    "need, in whatever order you judge useful; there is no required workflow.\n\n"
    "Skills:\n"
    "- observe: current FRONT RGB.\n"
    "- move: up to 3 of FORWARD(25-75 cm), TURN_LEFT/RIGHT(15-45 deg), magnitudes required.\n"
    "- stop: end the episode. Evidence is required by the skill.\n\n"
    "Looking is free. move spends one movement from a small budget. When you are done, "
    "or the budget is gone, call stop."
)

PROMPT_OPTIONAL = (
    "You are a robot carrying out an indoor navigation instruction. You act only through "
    "the skills listed below. Nothing is shown automatically. Choose whichever skills you "
    "need, in whatever order you judge useful; there is no required workflow.\n\n"
    "Skills:\n"
    "- observe: current FRONT RGB.\n"
    "- observe_panorama: LEFT/FRONT/RIGHT covering 360 degrees. Required before "
    "navigate_to, because pixel coordinates are relative to one of those views.\n"
    "- observe_local_map: recent RGB-D occupancy (agent-centered, facing up).\n"
    "- navigate_to: walk to a visible floor pixel in a named panorama view.\n"
    "- move: up to 3 of FORWARD(25-75 cm), TURN_LEFT/RIGHT(15-45 deg), magnitudes required.\n"
    "- go_back: walk back to a checkpoint recorded by navigate_to or move.\n"
    "- recall: remaining budget, recent outcomes, failed targets, checkpoints, any notes.\n"
    "- refine_instruction / note_progress: optional bookkeeping. Skip them if you do not need them.\n"
    "- stop: end the episode. Evidence is required by the skill.\n\n"
    "Looking and remembering are free. navigate_to, move, and go_back each spend one "
    "movement from a small budget. When you are done, or the budget is gone, call stop."
)

PROMPT_FORCED = (
    "You are a robot carrying out an indoor navigation instruction. You act only through "
    "your skills. Perception is on demand: nothing is shown automatically.\n\n"
    "Skills:\n"
    "- observe: current FRONT RGB, when a forward check is enough.\n"
    "- observe_panorama: LEFT/FRONT/RIGHT covering 360 degrees, when you need the "
    "surrounding environment or a pixel for navigate_to.\n"
    "- observe_local_map: recent RGB-D occupancy, when you need nearby free vs blocked floor.\n"
    "- navigate_to: walk to a visible floor pixel in a named panorama view.\n"
    "- move: up to 3 of FORWARD(25-75 cm), TURN_LEFT/RIGHT(15-45 deg), magnitudes required.\n"
    "- go_back: walk back to a checkpoint recorded by navigate_to or move.\n"
    "- recall: remaining budget, recent outcomes, failed targets, checkpoints, notes.\n"
    "- refine_instruction: split the instruction into ordered subtasks.\n"
    "- note_progress: record a finished subtask with visual evidence.\n"
    "- stop: end the episode. Evidence is required by the skill.\n\n"
    "Work in this order. Start by calling refine_instruction. Then loop: look only if you "
    "are unsure, walk, check what changed. Call note_progress with real visual evidence "
    "each time you finish a subtask. If a movement fails or the scene does not match "
    "what you expected, call recall() and go_back(checkpoint_id) rather than pushing "
    "further in the wrong direction. Looking and remembering are free. navigate_to, "
    "move, and go_back each spend one movement from a small budget. When the instruction "
    "is genuinely complete, call stop with the evidence; if you run out of budget, call "
    "stop and say plainly that you did not arrive."
)

_MODE_TABLE = {
    "minimal": {
        "tools": MINIMAL_SKILLS,
        "system_prompt": PROMPT_MINIMAL,
        "budget_skills": ("move",),
    },
    "optional": {
        "tools": OPTIONAL_SKILLS,
        "system_prompt": PROMPT_OPTIONAL,
        "budget_skills": ("navigate_to", "move", "go_back"),
    },
    "forced": {
        "tools": FORCED_SKILLS,
        "system_prompt": PROMPT_FORCED,
        "budget_skills": ("navigate_to", "move", "go_back"),
    },
}


def resolve_mode(name):
    mode = str(name or "optional").strip().lower()
    if mode not in _MODE_TABLE:
        raise ValueError("skill_mode must be one of %s, got %r" % (list(MODES), name))
    return mode


def mode_config(name):
    mode = resolve_mode(name)
    config = dict(_MODE_TABLE[mode])
    config["mode"] = mode
    return config


def count_skills(transcript):
    return dict(Counter(str(item.get("skill") or "") for item in transcript or []))


def usage_flags(counts):
    """Per-episode booleans: did the model invoke each extra skill at least once."""
    counts = counts or {}
    return {skill: int(counts.get(skill, 0) > 0) for skill in EXTRA_SKILLS}


def _mean(values):
    values = list(values)
    return (sum(values) / len(values)) if values else 0.0


def summarize_records(records):
    """SR/SPL plus whether extra skills were actually used."""
    records = list(records or [])
    metric_keys = sorted({k for r in records for k, v in r.get("metrics", {}).items()
                          if isinstance(v, (int, float))})
    usage_rate = {
        skill: _mean((r.get("used_extra_skill") or {}).get(skill, 0) for r in records)
        for skill in EXTRA_SKILLS
    }
    count_keys = sorted({k for r in records for k in (r.get("skill_counts") or {})})
    return {
        "episodes": len(records),
        "averages": {
            k: _mean(r["metrics"][k] for r in records if k in r.get("metrics", {}))
            for k in metric_keys
        },
        "mean_moves": _mean(r.get("moves_used", 0) for r in records),
        "used_any_extra": _mean(
            int(any((r.get("used_extra_skill") or {}).values())) for r in records),
        "skill_usage_rate": usage_rate,
        "mean_skill_counts": {
            skill: _mean((r.get("skill_counts") or {}).get(skill, 0) for r in records)
            for skill in count_keys
        },
    }


def comparison_row(mode, summary):
    averages = summary.get("averages") or {}
    usage = summary.get("skill_usage_rate") or {}
    return {
        "mode": mode,
        "episodes": summary.get("episodes"),
        "success": averages.get("success"),
        "spl": averages.get("spl"),
        "oracle_success": averages.get("oracle_success"),
        "distance_to_goal": averages.get("distance_to_goal"),
        "mean_moves": summary.get("mean_moves"),
        "used_any_extra": summary.get("used_any_extra"),
        "used_refine": usage.get("refine_instruction"),
        "used_note": usage.get("note_progress"),
        "used_go_back": usage.get("go_back"),
        "used_panorama": usage.get("observe_panorama"),
        "used_local_map": usage.get("observe_local_map"),
        "used_navigate_to": usage.get("navigate_to"),
        "used_recall": usage.get("recall"),
    }
