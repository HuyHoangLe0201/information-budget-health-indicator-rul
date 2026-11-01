"""Generate refs/bib.tex (thebibliography) in order of first \\cite in main.tex.
Fails on a cited key that is missing from the database; warns on unused entries."""
import re, sys
sys.path.insert(0, '.')
from refs_db import R

tex = open('../main.tex', encoding='utf-8').read()
tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)           # strip comments
tex = re.sub(r'\\input\{refs/bib\}', '', tex)
keys = []
for m in re.finditer(r'\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        k = k.strip()
        if k and k not in keys: keys.append(k)
missing = [k for k in keys if k not in R]
if missing: sys.exit('MISSING from database: %s' % missing)
unused = [k for k in R if k not in keys]
if unused: print('unused database entries:', unused)

def fmt(k):
    r = R[k]; a = r['authors']; y = r['year']; t = r['title']; kind = r['kind']
    doi = r['doi']; tail = (' doi:%s' % doi) if doi else ''
    if kind == 'art':
        s = '%s %s %s {\\it %s}' % (a, y, t, r['venue'])
        if r['vol']: s += ' {\\bf %s}' % r['vol']
        if r['pages']: s += ' %s' % r['pages']
    elif kind == 'book':
        s = '%s %s {\\it %s} %s' % (a, y, t, r['venue'])
    elif kind == 'proc':
        s = '%s %s %s {\\it %s}' % (a, y, t, r['venue'])
        if r['vol']: s += ' {\\bf %s}' % r['vol']
        if r['pages']: s += ' pp %s' % r['pages']
    elif kind == 'arxiv':
        s = '%s %s %s %s' % (a, y, t, r['venue'])
    elif kind == 'data':
        s = '%s %s %s {\\it %s}' % (a, y, t, r['venue'])
    if r['extra']: tail += ' ' + r['extra']
    return '\\bibitem{%s} %s%s' % (k, s, tail)

out = ['\\begin{thebibliography}{%d}' % len(keys)] + [fmt(k) for k in keys] + ['\\end{thebibliography}']
open('bib.tex', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(len(keys), 'references written')
