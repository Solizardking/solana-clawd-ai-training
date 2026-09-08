"""Local, page-cited PDF/document retrieval; no remote uploads."""
import hashlib
import re
import sqlite3
from pathlib import Path


def build_index(paths, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target)
    db.execute("DROP TABLE IF EXISTS research")
    db.execute("CREATE VIRTUAL TABLE research USING fts5(source, page UNINDEXED, text)")
    seen, report = set(), []
    for path in paths:
        path = Path(path)
        files = sorted(path.glob('*.pdf')) if path.is_dir() else [path]
        for file in files:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            if file.suffix.lower() == '.pdf':
                from pypdf import PdfReader
            pages = ([(i + 1, page.extract_text() or '') for i, page in enumerate(PdfReader(file).pages)]
                     if file.suffix.lower() == '.pdf' else [(1, file.read_text())])
            chunks = 0
            for page, text in pages:
                for start in range(0, len(text), 1800):
                    chunk = text[start:start + 2200]
                    if chunk.strip():
                        db.execute('INSERT INTO research VALUES (?, ?, ?)', ('/'.join(file.parts[-2:]), page, chunk))
                        chunks += 1
            report.append(dict(source=file.name, sha256=digest, pages=len(pages), chunks=chunks))
    db.commit()
    db.close()
    return report


def search(target, query, limit=4):
    if not Path(target).exists():
        return []
    terms = re.findall(r'[A-Za-z0-9_]{3,}', query)[:24]
    if not terms:
        return []
    with sqlite3.connect(f'file:{Path(target).resolve()}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT source,page,text FROM research WHERE research MATCH ? ORDER BY rank LIMIT ?',
                          (' OR '.join('"' + t + '"' for t in terms), min(limit, 8))).fetchall()
    return [dict(source=s, page=p, text=t) for s, p, t in rows]
