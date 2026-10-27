
import asyncio,copy,json,os,sys
import gpt_vln.evidence_prompt as nav
runner=nav.runner
runner.nav.NAV=copy.deepcopy(runner.nav.NAV)
runner.nav.NAV['properties'].update({
 'history_panorama_ids':{'type':'array','maxItems':3,'items':{'type':'integer'}},
 'map_operation':{'type':'string','enum':['none','annotate_node','navigate_to_node']},
 'map_node_id':{'type':['integer','null']},
 'map_annotation':{'type':'string'},
 'target':{'type':'object','additionalProperties':False,'properties':{
  'description':{'type':'string'},'view':{'type':'string','enum':['front','right','rear','left']},
  'point_norm':{'type':['array','null'],'minItems':2,'maxItems':2,'items':{'type':'number'}},
  'status':{'type':'string','enum':['unknown','seen','approaching','possibly_passed','reached']}},
  'required':['description','view','point_norm','status']},
 'expected_effect':{'type':'object','additionalProperties':False,'properties':{
  'goal':{'type':'string'},'expected_distance_change':{'type':'string','enum':['decrease','stable','unknown']}},
  'required':['goal','expected_distance_change']},
 'entrance_candidates':{'type':'array','maxItems':3,'items':{'type':'object','additionalProperties':False,'properties':{
  'description':{'type':'string'},'view':{'type':'string','enum':['front','right','rear','left']},
  'bbox_norm':{'type':['array','null'],'minItems':4,'maxItems':4,'items':{'type':'number'}},
  'possible_destination':{'type':'string'},'instruction_relevance':{'type':'string','enum':['potentially_relevant','uncertain','unlikely']},
  'evidence_frame':{'type':'integer'}},'required':['description','view','bbox_norm','possible_destination','instruction_relevance','evidence_frame']}}})
runner.nav.NAV['required']+=['map_operation','map_node_id','map_annotation','history_panorama_ids','target','expected_effect','entrance_candidates']
runner.nav.PROMPT+='''
TARGET ANCHOR: In the same navigation decision, name the immediate visible
landmark and give its approximate point_norm [x,y] in the named ORIGINAL four-view
RGB tile (0..1), or null when unknown. This is a visual hypothesis, not a verified
object. State the expected effect of your action chunk. A local measured critic may
interrupt the remaining primitive actions if a bound anchor moves behind or measured
distance rises. It cannot certify door identity, arrival or the instruction route.
The default visual history is only the immediately previous decision's four-view
panorama. A relevant saved event panorama is added only after a local conflict.
history_panorama_ids is retained for schema compatibility; always return [].
Keep the target description short and point to a visible surface near the intended
opening, not an unobserved point beyond a door. Reassess after each executed chunk.
PREDICTED SEMANTIC ROUTE: Also list at most three visible doorway/opening candidates
in entrance_candidates, including candidates you do not plan to enter now.
Only list a real traversable-looking opening with a visible floor continuation;
closed cabinet doors, shower glass, furniture gaps and already traversed rooms
are not candidates. Prefer at most two instruction-relevant entrances. bbox_norm
is [xmin,ymin,xmax,ymax] on the named panorama tile, normalized 0..1; use null if
localization is unclear. possible_destination is a fallible visual guess; relevance
to the instruction is not confirmation. Keep candidates that remain unresolved in
the predicted graph in context. At most three are active; older candidates may
be archived and can be revisited from stored frames. Distinguish predicted edges from actually traversed
edges. Do not claim a candidate is the goal without route and image evidence.
HISTORICAL IMAGE OVERRIDE: The next decision receives one CURRENT composite image (four-view panorama plus observed map), the immediately previous decision's four-view panorama, and at most one event panorama only when a measured conflict requests it. No sampled FRONT frames are supplied. Return history_panorama_ids=[]. Cite visual evidence only from images actually supplied in this call.

MAP SENSOR OVERRIDE: This candidate receives an online observed RGB-D map
from four cameras. It shows measured colour cells and explicitly unknown
checkerboard space. This supersedes older RGB-only/no-depth language.
Use the map image and node table in each decision. The map is metric in the
simulator's own pose frame; it is not a target/goal map.
FOUR MAP SKILLS: The observed map context includes simulator-measured current self pose relative to start; use it for localization only, never as goal evidence. Each annotated node records the exact pose and heading where you named it, a four-view frame ID, your stage hypothesis, links, visits, return walked distance and relative turn. Use a stored frame ID in history_panorama_ids on the NEXT decision if its visual evidence matters. The return distance is along recorded history, not a shortest route.
FOUR MAP SKILLS: The observed_semantic_map is your observe_map result: visited nodes,
chronological executed edges and your own visual hypotheses. Unvisited space is unknown.
Topological nodes represent actually visited frames, not inferred doors or free space.
To label an existing node, set map_operation=annotate_node, map_node_id to its ID,
map_annotation to a short semantic label and cite visible evidence_frames. This saves
an unverified Claude annotation. To revisit an existing node, set
map_operation=navigate_to_node, map_node_id to its ID and maneuver=return_to_anchor.
The harness follows recorded route points back to the node through actual
Habitat actions, then returns new RGB and map on the next decision.
Use map_operation=none and map_node_id=null, map_annotation='' otherwise. You alone
choose all motion, revisions and STOP. Do not treat a nearby node pose as arrival.
At STOP, verify the full ordered route and the final relation using actual earlier
and current frames. A matching object alone is insufficient, especially where
several similar portraits or rooms exist. If your route_verdict remains unknown,
continue with a short grounded move or revisit a saved node to resolve it.
For "walk down the hallway until near" instructions, confirm that your executed
route entered and followed that hallway before treating a side-gallery object as
the destination. Your own place labels are hypotheses, not arrival evidence.

GT LOCALIZATION OVERRIDE: this run explicitly supplies gt_own_pose_map: real own
position relative to start, measured heading and height, and accumulated executed
trajectory points. This supersedes older RGB-only/no-true-pose descriptions.
Use measured own displacement/height to judge movement and climb direction, and
compare own position to a stored anchor for return. Map points are NOT free-space
or occupancy; never shortcut a trail through unseen space. A nearby anchor alone
does not verify route reentry. No goal coordinates, reference route, navmesh queries
or shortest-path planning are supplied. Exact second tread remains unverified by
height alone. Keep RGB landmark evidence, short batches, and bounded recovery.
'''
