import sqlite3
import glob
import datetime

conn = sqlite3.connect('panchayat.db')
conn.row_factory = sqlite3.Row

# Stats
row = conn.execute('SELECT * FROM sentiment_stats ORDER BY id DESC LIMIT 1').fetchone()
print('=== SENTIMENT STATS ===')
print(dict(row) if row else 'EMPTY')

# Hourly trends range
r = conn.execute('SELECT MIN(time), MAX(time), COUNT(*) as cnt FROM hourly_trends').fetchone()
print(f'=== HOURLY TRENDS: {r[2]} rows, range [{r[0]}] -> [{r[1]}]')

# Raw JSONL backlog
files = glob.glob('data/raw/*.jsonl')
total = sum(1 for f in files for _ in open(f, encoding='utf-8'))
print(f'=== RAW BACKLOG: {total} unprocessed posts in {len(files)} file(s)')

# What period filter returns
ist_now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
for label, hours in [('1h', 1), ('6h', 6), ('1d', 24)]:
    cutoff = (ist_now - datetime.timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M')
    cnt = conn.execute('SELECT COUNT(*) FROM hourly_trends WHERE time >= ?', (cutoff,)).fetchone()[0]
    print(f'=== TRENDS IN LAST {label} (cutoff {cutoff}): {cnt} rows')

conn.close()
