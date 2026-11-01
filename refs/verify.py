"""Verify citations against Crossref: prints the best match for (title query, expected first-author surname, expected year) with a similarity score."""
import json, sys, urllib.parse, subprocess, difflib, re, time
def crossref(q, rows=5):
    url = 'https://api.crossref.org/works?rows=%d&select=DOI,title,author,container-title,volume,issue,page,issued,type,publisher&query.bibliographic=%s' % (rows, urllib.parse.quote(q))
    out = subprocess.run(['curl', '-sS', '--cacert', '/root/.ccr/ca-bundle.crt', '-m', '40', url], capture_output=True, text=True).stdout
    try: return json.loads(out)['message']['items']
    except Exception: return []
def fmt(it):
    au = it.get('author', []); names = ', '.join((a.get('family', '?') + ' ' + ''.join(w[0] for w in a.get('given', '').replace('-', ' ').split())) for a in au[:12]) + (' et al' if len(au) > 12 else '')
    y = (it.get('issued', {}).get('date-parts', [[None]])[0] or [None])[0]
    return dict(doi=it.get('DOI'), title=(it.get('title') or [''])[0], authors=names, first=(au[0].get('family', '') if au else ''), journal=(it.get('container-title') or [''])[0], vol=it.get('volume'), issue=it.get('issue'), pages=it.get('page'), year=y, type=it.get('type'))
def check(key, title, surname, year):
    best = None
    for it in crossref(title + ' ' + surname):
        f = fmt(it); sim = difflib.SequenceMatcher(None, re.sub(r'\W+', ' ', title.lower()), re.sub(r'\W+', ' ', f['title'].lower())).ratio()
        ok = surname.lower() in f['authors'].lower()
        sc = sim + (0.3 if ok else 0) + (0.1 if f['year'] == year else 0)
        if best is None or sc > best[0]: best = (sc, sim, ok, f)
    return best
if __name__ == '__main__':
    items = json.load(open(sys.argv[1])); res = {}
    for key, title, surname, year in items:
        b = check(key, title, surname, year); time.sleep(0.3)
        if b is None: print(f'[{key}] NO RESULT'); continue
        sc, sim, ok, f = b; flag = 'OK ' if sim > 0.9 and ok else 'CHECK'
        print(f"[{key}] {flag} sim={sim:.2f} author={'Y' if ok else 'N'} year_expected={year}\n    {f['authors']} ({f['year']}) {f['title']}. {f['journal']} {f['vol']}({f['issue']}) {f['pages']} doi:{f['doi']} [{f['type']}]")
        res[key] = f
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
