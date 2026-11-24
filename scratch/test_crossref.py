import urllib.request, json
url = 'https://api.crossref.org/works/10.1016/j.inffus.2022.03.007'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.load(resp)
item = data['message']
print('Title:', item['title'][0])
print('Authors:', ' and '.join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get('author', [])]))
print('Journal:', item.get('container-title', [''])[0])
print('Volume:', item.get('volume'), 'Pages:', item.get('page'), 'Year:', item.get('issued', {}).get('date-parts', [[None]])[0][0])
print('DOI:', item['DOI'])
