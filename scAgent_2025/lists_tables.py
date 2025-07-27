import psycopg2

conn = psycopg2.connect(
    host="10.28.1.24",
    port=5432,
    user="yanglab",
    password="labyang",
    dbname="scagent"
)
cur = conn.cursor()
cur.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='BASE TABLE' ORDER BY table_schema, table_name;")
for schema, table in cur.fetchall():
    print(f"{schema}.{table}")
cur.close()
conn.close()