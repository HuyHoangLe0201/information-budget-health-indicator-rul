import sys, json
from verify import crossref, fmt
for q in sys.argv[1:]:
    print('##', q)
    for it in crossref(q, rows=5):
        f = fmt(it); print('  ', f['authors'][:60], '(%s)' % f['year'], f['title'][:90], '|', f['journal'][:40], f['vol'], f['issue'], f['pages'], f['doi'], f['type'])
