
import asyncio,json,os,sys,copy
import gpt_vln.review_prompt as navigation
from gpt_vln.evidence_check import SCHEMA,PROMPT
runner=navigation.runner
runner.nav.NAV=copy.deepcopy(runner.nav.NAV)
runner.nav.NAV['properties']['route_evidence']=SCHEMA
runner.nav.NAV['required'].append('route_evidence')
runner.nav.PROMPT+='\n'+PROMPT
