"""Find exact DOIs for papers using Crossref Title Search."""

import urllib.request
import urllib.parse
import json

QUERIES = [
    ("kumar2015image", "Image fusion based on pixel significance using cross bilateral filter"),
    ("petrovic_xydeas", "Objective image fusion performance measure Petrovic Xydeas"),
    ("petrovic2005", "Objective evaluation of hybrid image fusion performance"),
    ("zhao2023ddfm", "DDFM Denoising Diffusion Model for Multi-Modality Image Fusion"),
    ("nejati2015", "A fast image fusion method based on multi-scale transform and spatial frequency Nejati"),
    ("tang2022piafusion", "Image fusion in the dark: A progressive infrared and visible image fusion network"),
    ("toet2014tno", "TNO Image Fusion Dataset Alexander Toet"),
]

def search_title(key, title):
    url = f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(title)}&rows=3"
    req = urllib.request.Request(url, headers={"User-Agent": "AntigravityResearchVerification/1.0 (mailto:verify@research.local)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            print(f"\n--- {key} ---")
            for item in items:
                doi = item.get("DOI")
                item_title = item.get("title", [""])[0] if item.get("title") else ""
                container = item.get("container-title", [""])[0] if item.get("container-title") else ""
                year = ""
                if "published-print" in item:
                    year = item["published-print"]["date-parts"][0][0]
                elif "published-online" in item:
                    year = item["published-online"]["date-parts"][0][0]
                print(f"  DOI: {doi}")
                print(f"  Title: {item_title}")
                print(f"  Container: {container} ({year})")
    except Exception as e:
        print(f"Error {key}: {e}")

if __name__ == "__main__":
    for key, title in QUERIES:
        search_title(key, title)
