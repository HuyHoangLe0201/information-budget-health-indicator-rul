"""Generate refs/bib.tex (thebibliography) in order of first \\cite in main.tex.
Fails on a cited key that is missing from the database; warns on unused entries."""
import re, sys
sys.path.insert(0, '.')
from refs_db import R

SRC = sys.argv[1] if len(sys.argv) > 1 else '../main.tex'; OUT = sys.argv[2] if len(sys.argv) > 2 else 'bib.tex'
tex = open(SRC, encoding='utf-8').read()
tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)           # strip comments
tex = re.sub(r'\\input\{[^}]*bib[^}]*\}', '', tex)
keys = []
for m in re.finditer(r'\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        k = k.strip()
        if k and k not in keys: keys.append(k)
missing = [k for k in keys if k not in R]
if missing: sys.exit('MISSING from database: %s' % missing)
unused = [k for k in R if k not in keys]
if unused: print('unused database entries:', unused)

INIT = re.compile(r'^[A-Z](-[A-Z])?$')
def person(item):
    toks = item.split()
    i = len(toks)
    while i > 1 and INIT.match(toks[i-1]): i -= 1
    if i == len(toks): return item                     # organisation or no initials
    ini = ''.join((t.split('-')[0] + '.-' + t.split('-')[1] + '.') if '-' in t else t + '.' for t in toks[i:])
    return ini + ' ' + ' '.join(toks[:i])
def authors_els(a):
    etal = a.endswith(' et al'); a = a[:-6] if etal else a
    items = [x.strip() for x in a.replace(' and ', ', ').split(', ') if x.strip()]
    s = ', '.join(person(x) for x in items)
    return s + (', et al.' if etal else '')
def pages_els(p):
    m = re.match(r'^(\d+)--(\d+)$', p)
    if m and len(m.group(2)) < len(m.group(1)): return m.group(1) + '--' + m.group(1)[:len(m.group(1)) - len(m.group(2))] + m.group(2)
    return p
def place(v):
    m = re.search(r'\s*\((.*?): (.*)\)', v)
    return (v[:m.start()] + (', ' if v[:m.start()].strip() else '') + m.group(2) + ', ' + m.group(1)) if m else v
def fmt(k):
    r = R[k]; a = authors_els(r['authors']); y = r['year']; t = r['title']; kind = r['kind']; v = r['venue']
    doi = (' \\url{https://doi.org/%s}' % r['doi']) if r['doi'] else ''
    extra = (' ' + r['extra']) if r['extra'] else ''
    if kind == 'art':
        s = '%s, %s, %s' % (a, t, v)
        if r['vol']: s += ' %s' % r['vol']
        s += ' (%s)' % y
        if r['pages']: s += ' %s' % pages_els(r['pages'])
        s += '.'
    elif kind == 'book':
        t2 = re.sub(r' (\d+)(nd|rd|th|st) edn$', r', \1\2 ed.', t); s = '%s, {\\it %s}, %s, %s.' % (a, t2, place(v), y)
    elif kind == 'proc':
        s = '%s, %s, in: %s' % (a, t, v)
        if r['vol']: s += ', vol. %s' % r['vol']
        s += ', %s' % y
        if r['pages']: s += ', pp. %s' % pages_els(r['pages'])
        s += '.'
    elif kind == 'arxiv':
        s = '%s, %s, %s (%s).' % (a, t, v, y)
    elif kind == 'data':
        s = '%s, %s, %s, %s.' % (a, t, place(v), y)
    return '\\bibitem{%s} %s%s%s' % (k, s, doi, extra)

out = ['\\begin{thebibliography}{%d}' % len(keys)] + [fmt(k) for k in keys] + ['\\end{thebibliography}']
open(OUT, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(len(keys), 'references written')
