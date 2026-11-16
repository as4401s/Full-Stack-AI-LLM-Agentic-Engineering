-- schema.sql - two small CareDesk tables for the SQL refresher (synthetic data only).
-- Works in PostgreSQL and in SQLite (the lab falls back to SQLite when PostgreSQL is not running).
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE appointments (
    id            INTEGER PRIMARY KEY,
    department_id INTEGER NOT NULL REFERENCES departments (id),
    patient_code  TEXT    NOT NULL,        -- a synthetic code, never a real patient
    scheduled_for TIMESTAMP NOT NULL,
    wait_minutes  INTEGER NOT NULL,        -- minutes the patient waited
    status        TEXT    NOT NULL         -- 'seen' or 'no_show'
);
