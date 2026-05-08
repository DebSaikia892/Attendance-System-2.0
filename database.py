import sqlite3

conn = sqlite3.connect("E:\\Projects\\Attendance System 2.0\\faces.db")
cursor = conn.cursor()
cursor.execute('select ID, Name from faces')
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.commit()
conn.close()