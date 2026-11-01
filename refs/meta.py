import sys, re, subprocess, html
def meta(url):
    t = subprocess.run(['curl', '-sSL', '--cacert', '/root/.ccr/ca-bundle.crt', '-m', '40', '-A', 'Mozilla/5.0', url], capture_output=True, text=True).stdout
    tags = re.findall(r'<meta\s+name="(citation_[a-z_]+)"\s+content="([^"]*)"', t) + [(b, a) for a, b in re.findall(r'<meta\s+content="([^"]*)"\s+name="(citation_[a-z_]+)"', t)]
    d = {}
    for k, v in tags: d.setdefault(k, []).append(html.unescape(v.strip()))
    return d
for u in sys.argv[1:]:
    d = meta(u); print('##', u)
    for k in ['citation_title', 'citation_author', 'citation_publication_date', 'citation_date', 'citation_online_date', 'citation_journal_title', 'citation_conference_title', 'citation_volume', 'citation_issue', 'citation_firstpage', 'citation_lastpage', 'citation_doi', 'citation_arxiv_id']:
        if k in d: print('  ', k.replace('citation_', ''), ':', '; '.join(d[k])[:300])
