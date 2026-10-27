
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated

from gpt_vln.route_prompt import runner

PROMPT = re.sub(r'HISTORICAL IMAGE OVERRIDE:.*?(?=MAP SENSOR OVERRIDE:)',
                '', runner.nav.PROMPT, flags=re.DOTALL)
PROMPT = PROMPT.replace(
    'You alone navigate an RGB-only indoor robot following the original instruction.',
    'You navigate an indoor robot following the original instruction.')
PROMPT = PROMPT.replace(
    'Current PANORAMA shows simultaneous FRONT0, RIGHT-90, REAR180, LEFT+90 views. History is at most8 uniformly sampled past FRONT images; their side/rear views are not supplied. Historical images retain old commanded heading.',
    'Current image is one labeled four-view panorama: FRONT, RIGHT, REAR, LEFT. Retrieved historical panoramas retain their old commanded heading.')
PROMPT = PROMPT.replace(
    'RGB relative pose has unknown metric scale; rotation is not translation or route recovery.',
    'Measured own pose gives metric motion; a turn is not translation or route recovery.')
PROMPT = PROMPT.replace(
    'No depth, simulator pose, reference path, navmesh, goal metrics or trained navigator.',
    'Own pose and RGB-D observed mapping are available; hidden goal position, reference path and navmesh are not.')
PROMPT = PROMPT.replace(
    'Four simultaneous horizontal cameras already cover horizontal directions; No FLOOR4 is supplied.',
    'The current four-view panorama covers horizontal directions; no FLOOR4 is supplied.')
PROMPT = PROMPT.replace('The default visual history is only the immediately previous decision\'s four-view\npanorama. A relevant saved event panorama is added only after a local conflict.\n', '')
PROMPT = PROMPT.replace('history_panorama_ids is retained for schema compatibility; always return [].', '')
PROMPT = PROMPT.replace(
    'Use a stored frame ID in history_panorama_ids on the NEXT decision if its visual evidence matters.',
    'Use a memory retrieval tool in this decision if a stored node image matters.')
PROMPT = PROMPT.replace(
    'Use the map image and node table in each decision. The map is metric in the',
    'Use the structured map node table in each decision. The map is metric in the')
PROMPT += '''
MEMORY CONTRACT: The current labeled four-view panorama and OBSERVED RGB-D MAP
are your default visual evidence. Use the original instruction from the first
turn and measured motion. Historical descriptions and stage claims are hypotheses.
This is one continuous episode conversation. Earlier current panoramas remain
in the conversation; use them for recent comparisons. For older or uncertain
places, retrieve the archived four-view panorama with a memory tool.
The four tiles are separate 90-degree cameras arranged FRONT, RIGHT, REAR, LEFT.
Their borders are not continuous perspective, so do not infer a doorway or corridor
crosses a tile boundary without checking the individual views and robot heading.
If current evidence suffices, decide directly. Call retrieve_keyframes for a specific
uncertain landmark, branch, or stage relation; call retrieve_uniform_history
when the whole route or progress is unclear. Tools return actual old four-view
panoramas in this same decision. A previous decision image is available only
when its frame is in the keyframe index. Retrieved pictures are evidence, not
ground truth. Earlier episode frames already in this conversation may be cited;
retrieve an older frame when its visual detail is uncertain.
Return history_panorama_ids=[]; history is retrieved by tools, not preloaded.
An obstacle or low translation means blocked, never arrival. STOP only when
the ordered instruction and final location are supported by visual evidence.
The observed RGB-D map shows only measured coverage, visited numbered nodes,
the walked blue path, and your current red pose arrow; dark cells are unknown.
It is not an oracle floor plan and has no benchmark goal or room labels.
The observed semantic map uses fixed world x/z axes, so image/map up and left
are not your current front and left. For an already visited node, use its
turn_deg_left relative to the measured current heading and its recorded links
and return_walked_m. Choose map_operation=navigate_to_node only for a visited
node; Python executes the recorded connection and you reobserve afterward.
'''
