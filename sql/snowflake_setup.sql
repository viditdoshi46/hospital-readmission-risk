-- ============================================================
-- Snowflake setup + load (run when you spin up a free trial)
-- Mirrors the local DuckDB analysis so the project is "cloud-ready".
--
-- Pipeline:  local CSV  ->  AWS S3 (external stage)  ->  Snowflake table
-- Replace <...> placeholders with your account values.
-- ============================================================

-- 0) Warehouse / db / schema
CREATE WAREHOUSE IF NOT EXISTS ANALYTICS_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
CREATE DATABASE IF NOT EXISTS HEALTHCARE;
CREATE SCHEMA  IF NOT EXISTS HEALTHCARE.READMISSION;
USE WAREHOUSE ANALYTICS_WH;
USE SCHEMA HEALTHCARE.READMISSION;

-- 1) Landing table for the enriched encounters (from src/features.py)
CREATE OR REPLACE TABLE ENCOUNTERS (
    patient_nbr           NUMBER,
    race                  STRING,
    gender                STRING,
    age                   STRING,
    age_mid               NUMBER,
    admission_type_id     NUMBER,
    discharge_group       STRING,
    time_in_hospital      NUMBER,
    num_lab_procedures    NUMBER,
    num_procedures        NUMBER,
    num_medications       NUMBER,
    number_outpatient     NUMBER,
    number_emergency      NUMBER,
    number_inpatient      NUMBER,
    prior_visits          NUMBER,
    number_diagnoses      NUMBER,
    n_meds_changed        NUMBER,
    a1c_tested            NUMBER,
    a1c_high              NUMBER,
    med_changed           NUMBER,
    on_diabetes_med       NUMBER,
    service_use_tier      STRING,
    medical_specialty     STRING,
    insulin               STRING,
    metformin             STRING,
    readmit_lt30          NUMBER
);

-- 2a) OPTION A - load from an S3 external stage (AWS layer)
CREATE STORAGE INTEGRATION IF NOT EXISTS S3_INT
    TYPE = EXTERNAL_STAGE STORAGE_PROVIDER = 'S3' ENABLED = TRUE
    STORAGE_AWS_ROLE_ARN = '<your-iam-role-arn>'
    STORAGE_ALLOWED_LOCATIONS = ('s3://<your-bucket>/readmission/');

CREATE OR REPLACE FILE FORMAT CSV_FF
    TYPE = CSV FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1 NULL_IF = ('');

CREATE OR REPLACE STAGE READMIT_STAGE
    STORAGE_INTEGRATION = S3_INT
    URL = 's3://<your-bucket>/readmission/'
    FILE_FORMAT = CSV_FF;

COPY INTO ENCOUNTERS
FROM @READMIT_STAGE/enriched.csv
FILE_FORMAT = (FORMAT_NAME = CSV_FF)
ON_ERROR = 'CONTINUE'
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 2b) OPTION B - quick load straight from your laptop (no S3):
--   snowsql -a <acct> -u <user> -q "PUT file://data/processed/enriched.csv @%ENCOUNTERS; COPY INTO ENCOUNTERS ..."

-- 3) The same business questions as duckdb_analysis.sql --------------

-- Overall rate
SELECT COUNT(*) AS encounters,
       ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
FROM ENCOUNTERS;

-- By discharge disposition
SELECT discharge_group,
       COUNT(*) AS encounters,
       ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
FROM ENCOUNTERS
GROUP BY discharge_group
ORDER BY readmit_rate_pct DESC;

-- Highest-risk actionable segment (high utilizers by discharge)
SELECT discharge_group, service_use_tier,
       COUNT(*) AS encounters,
       ROUND(100*AVG(readmit_lt30),1) AS readmit_rate_pct
FROM ENCOUNTERS
WHERE service_use_tier = 'High (3+)'
GROUP BY discharge_group, service_use_tier
ORDER BY readmit_rate_pct DESC;

-- 4) A view Tableau/Streamlit can point at
CREATE OR REPLACE VIEW V_READMIT_BY_DISCHARGE AS
SELECT discharge_group,
       COUNT(*) AS encounters,
       AVG(readmit_lt30) AS readmit_rate
FROM ENCOUNTERS GROUP BY discharge_group;
