"""
Run the DuckDB analysis queries against data/processed/enriched.csv and print
the results. Proves the SQL layer works locally with zero cloud setup.

Run:
    python src/run_sql.py
"""
from pathlib import Path
import duckdb

from config import DATA_PROCESSED

ENRICHED = DATA_PROCESSED / "enriched.csv"

QUERIES = {
    "Overall 30-day readmission rate": """
        SELECT COUNT(*) AS encounters,
               ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
        FROM enc""",
    "Rate by discharge disposition": """
        SELECT discharge_group,
               COUNT(*) AS encounters,
               ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
        FROM enc GROUP BY discharge_group ORDER BY readmit_rate_pct DESC""",
    "Rate by utilization tier": """
        SELECT service_use_tier,
               COUNT(*) AS encounters,
               ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
        FROM enc GROUP BY service_use_tier ORDER BY readmit_rate_pct DESC""",
    "Highest-risk actionable segment (High utilizers x discharge)": """
        SELECT discharge_group, service_use_tier,
               COUNT(*) AS encounters,
               ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
        FROM enc WHERE service_use_tier='High (3+)'
        GROUP BY discharge_group, service_use_tier
        ORDER BY readmit_rate_pct DESC""",
}


def main() -> None:
    if not ENRICHED.exists():
        raise SystemExit("Run src/features.py first to create enriched.csv")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE enc AS SELECT * FROM read_csv_auto('{ENRICHED}', header=true)")
    for title, q in QUERIES.items():
        print(f"\n=== {title} ===")
        print(con.execute(q).df().to_string(index=False))


if __name__ == "__main__":
    main()
