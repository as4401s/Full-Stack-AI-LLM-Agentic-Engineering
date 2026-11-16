-- queries.sql - a join, a GROUP BY and a window function on the CareDesk tables.

-- 1. JOIN: put the department name next to each appointment.
SELECT a.id, d.name AS department, a.patient_code, a.wait_minutes
FROM appointments AS a
JOIN departments AS d ON d.id = a.department_id
WHERE a.id <= 6
ORDER BY a.id;

-- 2. GROUP BY: one row per department with the number of appointments and the average wait.
SELECT d.name AS department,
       COUNT(*)                            AS appointments,
       ROUND(AVG(a.wait_minutes), 1)       AS avg_wait,
       MAX(a.wait_minutes)                 AS longest_wait
FROM appointments AS a
JOIN departments AS d ON d.id = a.department_id
GROUP BY d.name
ORDER BY avg_wait DESC;

-- 3. WINDOW function: keep every row, and add a rank within the department and a running average.
SELECT d.name AS department,
       a.scheduled_for,
       a.wait_minutes,
       RANK() OVER (PARTITION BY a.department_id ORDER BY a.wait_minutes DESC) AS wait_rank,
       ROUND(AVG(a.wait_minutes) OVER (
             PARTITION BY a.department_id
             ORDER BY a.scheduled_for
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 1)              AS running_avg
FROM appointments AS a
JOIN departments AS d ON d.id = a.department_id
WHERE d.name = 'Radiology'
ORDER BY a.scheduled_for;
