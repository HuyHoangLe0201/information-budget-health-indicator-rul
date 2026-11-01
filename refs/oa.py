import subprocess, json, sys, time, urllib.parse
def get(url):
    for _ in range(3):
        out = subprocess.run(['curl', '-sS', '--cacert', '/root/.ccr/ca-bundle.crt', '-m', '40', url], capture_output=True, text=True).stdout
        try: return json.loads(out)
        except Exception: time.sleep(2)
    return None
def abstract(w):
    inv = w.get('abstract_inverted_index')
    if not inv: return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs: pos[i] = word
    return ' '.join(pos[i] for i in sorted(pos))
def search(q, n=6, sort='relevance_score:desc', filt=''):
    u = 'https://api.openalex.org/works?per-page=%d&sort=%s&search=%s%s&select=title,authorships,publication_year,primary_location,doi,cited_by_count,abstract_inverted_index,type,biblio' % (n, sort, urllib.parse.quote(q), ('&filter=' + filt) if filt else '')
    d = get(u); return (d or {}).get('results', [])
if __name__ == '__main__':
    for q in sys.argv[1:]:
        print('##', q)
        for w in search(q):
            au = ', '.join(a['author']['display_name'] for a in w['authorships'][:3]); src = ((w.get('primary_location') or {}).get('source') or {}).get('display_name'); b = w.get('biblio') or {}
            print(' -', w['title'][:110], '|', au, '|', w['publication_year'], '|', src, b.get('volume'), b.get('first_page'), '|', w.get('doi'), '| cites', w['cited_by_count'])
            a = abstract(w); print('     ', (a or '')[:330])
        time.sleep(1.2)
