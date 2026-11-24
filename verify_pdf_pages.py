import pypdf

reader = pypdf.PdfReader('paper/CADM_MSWN_v2_Conference_Paper.pdf')
print('Total pages in generated PDF:', len(reader.pages))
for i, p in enumerate(reader.pages):
    txt = p.extract_text()
    print(f'=== PAGE {i+1} ({len(txt)} chars) ===')
    lines = [line.strip() for line in txt.split('\n') if line.strip()]
    print('  First 3 lines:', lines[:3])
    print('  Last 2 lines:', lines[-2:])
