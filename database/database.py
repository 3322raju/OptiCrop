import sqlite3

conn = sqlite3.connect("database/history.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop TEXT,
    nitrogen REAL,
    phosphorus REAL,
    potassium REAL,
    temperature REAL,
    humidity REAL,
    ph REAL,
    rainfall REAL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")