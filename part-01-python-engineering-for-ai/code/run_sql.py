"""run_sql.py - run the schema, seed and queries from ../data and print the results.

What it does: builds the CareDesk tables, inserts the synthetic rows and prints each query result.
It uses SQLite (built into Python), so it runs anywhere with nothing to install. The statements are
standard SQL (join, GROUP BY, window function); the part README shows how to run the same files in PostgreSQL.
What it needs: nothing beyond Python.
What to expect: three result tables (join, group by, window).
Run it:  uv run python run_sql.py
"""
import sqlite3
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def run_queries() -> None:
    """Create the tables, load the data, run every query in queries.sql and print the rows."""
    con = sqlite3.connect(":memory:")
    con.executescript((DATA / "schema.sql").read_text())
    con.executescript((DATA / "seed.sql").read_text())
    for number, query in enumerate((DATA / "queries.sql").read_text().split(";"), start=1):
        sql = "\n".join(line for line in query.splitlines() if not line.strip().startswith("--")).strip()
        if not sql:
            continue
        cursor = con.execute(sql)
        header = [c[0] for c in cursor.description]
        print(f"--- query {number}")
        print(" | ".join(header))
        for row in cursor.fetchall():
            print(" | ".join(str(v) for v in row))


if __name__ == "__main__":
    run_queries()
