-- ============================================================
-- Hospital Readmission - SQL analysis (DuckDB, runs locally)
-- Run:  duckdb < sql/duckdb_analysis.sql
-- or via src/run_sql.py which loads data/processed/enriched.csv
-- These are the exact business questions the dashboard answers.
-- ============================================================

-- Load the enriched encounter table produced by src/features.py
CREATE OR REPLACE TABLE encounters AS
SELECT * FROM read_csv_auto('data/processed/enriched.csv', header=true);

-- 1) Overall 30-day readmission rate + volume
SELECT
    COUNT(*)                                  AS encounters,
    SUM(readmit_lt30)                         AS readmissions_30d,
    ROUND(100.0 * AVG(readmit_lt30), 1)       AS readmit_rate_pct
FROM encounters;

-- 2) Readmission rate by discharge disposition (the biggest lever)
SELECT
    discharge_group,
    COUNT(*)                            AS encounters,
    ROUND(100.0 * AVG(readmit_lt30),1)  AS readmit_rate_pct
FROM encounters
GROUP BY discharge_group
ORDER BY readmit_rate_pct DESC;

-- 3) Rate by prior-utilization tier (repeat visitors drive risk)
SELECT
    service_use_tier,
    COUNT(*)                            AS encounters,
    ROUND(100.0 * AVG(readmit_lt30),1)  AS readmit_rate_pct
FROM encounters
GROUP BY service_use_tier
ORDER BY readmit_rate_pct DESC;

-- 4) Rate by age band
SELECT
    age,
    COUNT(*)                            AS encounters,
    ROUND(100.0 * AVG(readmit_lt30),1)  AS readmit_rate_pct
FROM encounters
GROUP BY age
ORDER BY age;

-- 5) Does an A1C test / med change during stay relate to readmission?
SELECT
    a1c_high,
    med_changed,
    COUNT(*)                            AS encounters,
    ROUND(100.0 * AVG(readmit_lt30),1)  AS readmit_rate_pct
FROM encounters
GROUP BY a1c_high, med_changed
ORDER BY readmit_rate_pct DESC;

-- 6) Highest-risk actionable segment: high utilizers discharged to SNF/Rehab
SELECT
    discharge_group,
    service_use_tier,
    COUNT(*)                            AS encounters,
    ROUND(100.0 * AVG(readmit_lt30),1)  AS readmit_rate_pct
FROM encounters
WHERE service_use_tier = 'High (3+)'
GROUP BY discharge_group, service_use_tier
ORDER BY readmit_rate_pct DESC;
