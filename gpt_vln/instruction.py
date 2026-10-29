
import re


_SEGMENT = re.compile(
    r"\bgo past (?P<landmark>[^,.;]+),\s*turn (?P<direction>left|right) "
    r"at (?P<trigger>[^,.;]+?) and (?P<finish>wait|stop) there\b",
    re.IGNORECASE,
)


def compound_segment(instruction):
    match = _SEGMENT.search(instruction)
    if match is None:
        return None
    return {
        'text': match.group(0),
        'landmark': match.group('landmark').strip(),
        'trigger': match.group('trigger').strip(),
        'turn': match.group('direction').lower(),
        'finish': match.group('finish').lower(),
        'relation': 'The door must be identified in relation to passing the landmark. '
                    'Keep candidate doors unresolved until their route relation is checked.',
    }


def instruction_constraints(instruction):

    turns = []
    for match in re.finditer(r'\bturn\s+(left|right)\b', instruction, re.IGNORECASE):
        turns.append({'order': len(turns), 'direction': match.group(1).lower(),
                      'source': instruction[match.start():match.end()]})
    completion = re.search(r'\b(wait|stop)\s+there\b', instruction, re.IGNORECASE)
    segment = compound_segment(instruction)
    return {
        'source': 'original instruction; directional words are immutable',
        'explicit_turns_in_order': turns,
        'linked_landmark_turn': segment,
        'completion': completion.group(0) if completion else None,
        'unknowns': 'which visible object matches an instruction noun; when a turn is due',
        'authority': 'Visual evidence may update object identity and progress, not rewrite an explicit turn.',
    }
