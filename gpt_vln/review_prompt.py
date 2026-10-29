
import asyncio,json,os,sys
import gpt_vln.base_prompt as runner
runner.nav.PROMPT+='''
PERIODIC ROUTE REVIEW OVERRIDE:
In addition to uniform8 past FRONT images you receive explicitly labeled saved KEY
horizontal panoramas. Use these for door/junction/stair evidence and return targets.
The periodic visual route reviewer runs every ten COMPLETED decisions, before the
next navigation decision. Its latest judgment has an age: it is fallible and can be
stale, never a command, STOP veto, independent ground truth or proof of completion.
Code rules still run each primitive. You alone choose all actions and STOP.
Before an ambiguous doorway choose a real earlier entrance/junction anchor. Preserve
it until the branch is resolved. Events and stage changes record hypotheses with
frames; never promote these labels alone to confirmed crossings.
When review is unknown, compare current panorama with supplied key images and the
actual chronological return_trail. Missing landmarks alone are not deviation.
If needed, choose a short visually supported segment toward the saved anchor using
return_to_anchor. Never replay inverse commands, invent graph edges or move straight
through walls toward an imagined coordinate. If the next safe segment cannot be
grounded, explicitly report that limitation rather than claim a successful return.
The existing panorama covers all directions: no scan-only turns. Return remains
unverified until current scene AND original route relations support reentry. A similar
view or restored motion alone does not suffice. Preserve the six-decision return bound.
'''
