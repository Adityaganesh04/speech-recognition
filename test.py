import sqlite3

c = sqlite3.connect('data/db/meetings.db')
row = c.execute("SELECT summary, intents FROM intelligence WHERE meeting_id='test_25_5146'").fetchone()

if row:
    with open('data/result.txt', 'w', encoding='utf-8') as f:
        f.write(f"SUMMARY: {row[0]}\n\nINTENTS: {row[1]}")
    print("SAVED!")
else:
    print("NO DATA FOUND")
