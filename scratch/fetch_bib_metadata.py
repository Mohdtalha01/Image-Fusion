import urllib.request
import json

queries = [
    ("ADF", "https://api.crossref.org/works?query.title=Fusion+of+Infrared+and+Visible+Sensor+Images+Based+on+Anisotropic+Diffusion+and+Karhunen-Loeve+Transform&rows=1"),
    ("Li2013", "https://api.crossref.org/works/10.1109/TIP.2013.2244222"),
    ("PIAFusion", "https://api.crossref.org/works/10.1016/j.inffus.2022.01.001"),
    ("DDFM", "https://api.crossref.org/works/10.1109/iccv51070.2023.00742"),
    ("CDDFuse", "https://api.crossref.org/works?query.title=CDDFuse:+Correlation-Driven+Dual-Branch+Feature+Decomposition+for+Multi-Modality+Image+Fusion&rows=1"),
    ("Han2013", "https://api.crossref.org/works?query.title=A+new+image+fusion+performance+metric+based+on+visual+information+fidelity&rows=1")
]

for name, url in queries:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Research/1.0)'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        if 'items' in data.get('message', {}):
            item = data['message']['items'][0]
        else:
            item = data['message']
        title = item.get('title', [''])[0] if isinstance(item.get('title'), list) else item.get('title')
        doi = item.get('DOI')
        container = item.get('container-title', [''])[0] if isinstance(item.get('container-title'), list) else item.get('container-title')
        authors = ' and '.join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get('author', [])])
        year = item.get('issued', {}).get('date-parts', [[None]])[0][0]
        vol = item.get('volume')
        issue = item.get('issue')
        pages = item.get('page')
        print(f"[{name}]")
        print(f"  Title: {title}")
        print(f"  Authors: {authors}")
        print(f"  Journal/Book: {container}")
        print(f"  Year: {year}, Vol: {vol}, Issue: {issue}, Pages: {pages}")
        print(f"  DOI: {doi}")
    except Exception as e:
        print(f"[{name}] Error: {e}")
