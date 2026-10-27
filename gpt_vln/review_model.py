
import asyncio
import json
import os
import sys

SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'route': {'type': 'string', 'enum': ['supported', 'contradicted', 'unknown']},
    'evidence': {'type': 'string'},
    'frames': {'type': 'array', 'maxItems': 10, 'items': {'type': 'integer'}},
    'suggested_return_anchor_frame': {'type': ['integer', 'null']},
    'next_check': {'type': 'string'},
    'task_summary': {'type': 'string'},
    'progress_hypothesis': {'type': 'string'},
    'direction_assessment': {'type': 'string', 'enum': ['supported', 'possibly_wrong', 'unknown']},
    'prediction_errors': {'type': 'array', 'maxItems': 3, 'items': {'type': 'string'}},
    'stage_progress': {'type': 'string'},
    'route_error_since_previous_review': {'type': 'string',
                                          'enum': ['no_error_found', 'possible_wrong_branch', 'confirmed_wrong_branch', 'unknown']},
}, 'required': ['route', 'evidence', 'frames', 'suggested_return_anchor_frame',
                'next_check', 'task_summary', 'progress_hypothesis',
                'direction_assessment', 'prediction_errors', 'stage_progress',
                'route_error_since_previous_review']}

PROMPT = '''Review the CURRENT ten-decision window against the full original
instruction and explicitly compare its start/end with the PREVIOUS REVIEW END.
The contact sheet is chronological and each tile is a four-view panorama. Read
the previous summary, actual measured motion, target hypotheses, and current
views. Distinguish seeing, approaching, passing and reaching each landmark. A
blocked forward action is not proof of arrival.
Identify prediction errors, possible wrong branches and unresolved identity.
You MUST state stage_progress for this window and route_error_since_previous_review:
whether the route became a possible or confirmed wrong branch since the prior
review. If the prior review said a side branch was unresolved, check whether the
new window actually entered it; continuing straight is not progress evidence.
Prior model stage/event labels are hypotheses, not verified progress. If
visual evidence is insufficient, mark unknown. Never use hidden GT goal,
reference path, navmesh or room annotations. Produce a concise task_summary
for ordinary future navigation turns: supported progress, uncertain claims,
current route hypothesis, and next visual check. Do not choose actions or
force STOP. Cite only supplied frame IDs.'''
