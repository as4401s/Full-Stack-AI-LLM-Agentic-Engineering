-- seed.sql - twelve synthetic appointments, small enough to check every answer by hand.
INSERT INTO departments (id, name) VALUES (1, 'Emergency'), (2, 'Cardiology'), (3, 'Radiology');

INSERT INTO appointments (id, department_id, patient_code, scheduled_for, wait_minutes, status) VALUES
 ( 1, 1, 'SYN-0001', '2026-03-02 08:00', 35, 'seen'),
 ( 2, 1, 'SYN-0002', '2026-03-02 09:30', 50, 'seen'),
 ( 3, 1, 'SYN-0003', '2026-03-02 11:00', 20, 'seen'),
 ( 4, 1, 'SYN-0004', '2026-03-02 13:00', 65, 'seen'),
 ( 5, 2, 'SYN-0005', '2026-03-02 08:30', 15, 'seen'),
 ( 6, 2, 'SYN-0006', '2026-03-02 10:00', 30, 'seen'),
 ( 7, 2, 'SYN-0007', '2026-03-02 14:00', 25, 'no_show'),
 ( 8, 3, 'SYN-0008', '2026-03-02 08:15', 40, 'seen'),
 ( 9, 3, 'SYN-0009', '2026-03-02 09:15', 10, 'seen'),
 (10, 3, 'SYN-0010', '2026-03-02 10:15', 20, 'seen'),
 (11, 3, 'SYN-0011', '2026-03-02 11:15', 30, 'seen'),
 (12, 3, 'SYN-0012', '2026-03-02 12:15', 25, 'seen');
