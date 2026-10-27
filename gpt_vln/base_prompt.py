
import asyncio
import json
import os
import sys
import gpt_vln.spatial_prompt as nav
nav.PROMPT = """You alone navigate an RGB-only indoor robot following the original instruction.
Current PANORAMA shows simultaneous FRONT0, RIGHT-90, REAR180, LEFT+90 views. History is at most8 uniformly sampled past FRONT images; their side/rear views are not supplied. Historical images retain old commanded heading.
Use the typed batch contract below, not legacy scalar turn/distance outputs.
Evidence ledger is the single persistent ordered-route memory. Cite only actual frames.
seen/reached/passed are different; commands do not prove outcomes. Disappearance is not
passing; a similar object is not the target without route and relational evidence.
RGB relative pose has unknown metric scale; rotation is not translation or route recovery.
Independent online critic is fallible evidence; choose actions yourself. At STOP review
inspect final landmark, requested relation and preceding route; then stop or move.
Choose short progress steps, not scans. Bound recovery; avoid repeating failed directions.
No depth, simulator pose, reference path, navmesh, goal metrics or trained navigator.
place<=12words, note<=50words describing evidence and expected post-action visual change.
"""
from gpt_vln.action_contract import CONTRACT, schema
nav.PROMPT += '\n' + CONTRACT.replace('FLOOR4 does too.', 'No FLOOR4 is supplied.').replace('FLOOR4 tiles','current horizontal panorama tiles')
nav.PROMPT += '\nNo downward camera is provided. Climb requires visible stair RGB evidence; do not claim inspection of absent down views. Uniform8 is fixed; an unsampled return anchor is only a textual reference, not a visible anchor.'
nav.NAV = schema(nav.NAV)
