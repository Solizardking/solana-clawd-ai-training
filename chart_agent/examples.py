"""Retrieve only training-split examples; held-out questions never enter RAG."""
import json
import re
import sqlite3
from pathlib import Path


def index_examples(dataset, target):
    with sqlite3.connect(target) as db:
        db.execute('DROP TABLE IF EXISTS examples')
        db.execute('CREATE VIRTUAL TABLE examples USING fts5(question, answer, source UNINDEXED)')
        count = 0
        with Path(dataset).open() as file:
            for line in file:
                row = json.loads(line)
                if row['split'] != 'train' or row['lane'] != 'chart_reasoning':
                    continue
                question = next(m['content'] for m in row['messages'] if m['role'] == 'user')
                answer = row['messages'][-1]['content']
                db.execute('INSERT INTO examples VALUES (?,?,?)', (question[:6000], answer[:6000], row['source'] + '#' + row['group']))
                count += 1
    return count


def retrieve(target, query):
    if not Path(target).exists():
        return []
    words = re.findall(r'[A-Za-z]{4,}', query)[:12]
    if not words:
        return []
    try:
        with sqlite3.connect(f'file:{Path(target).resolve()}?mode=ro', uri=True) as db:
            rows = db.execute('SELECT question,answer,source FROM examples WHERE examples MATCH ? ORDER BY rank LIMIT 1',
                              (' OR '.join('"' + w + '"' for w in words),)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [dict(question=q[:700], answer=a[:1200], source=s, kind='training_example_not_current_evidence') for q, a, s in rows]
