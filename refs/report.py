import sys; sys.path.insert(0,'.')
from refs_db import R
import re
tex=open('../main.tex',encoding='utf-8').read()
keys=[]
for m in re.finditer(r'\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}',tex):
    for k in m.group(1).split(','):
        k=k.strip()
        if k not in keys: keys.append(k)
vm={'C':'Crossref record','P':'publisher/proceedings page','A':'arXiv/PMLR/NeurIPS record','M':'author manuscript / full text','L':'library catalogue (Open Library)','W':'authoritative web listing'}
cm={'full':'full text read','abstract':'abstract read','title':'bibliographic record only (title/venue/pages); content not read','none':'classical source cited for a standard fact; content not re-read'}
out=['# Reference verification log','','Order = first citation in main.tex. "Metadata" = how bibliographic data were verified; "Content" = how much of the source was read for the claim it supports.','','| # | Key | Metadata | Content |','|---|---|---|---|']
for i,k in enumerate(keys,1):
    r=R[k]; out.append('| %d | %s | %s | %s |'%(i,k,vm[r['v']],cm[r['content']]))
open('VERIFICATION.md','w').write('\n'.join(out)+'\n'); print(len(keys))
