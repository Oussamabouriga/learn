SELECT 
  FLOOR(
    TIMESTAMPDIFF(HOUR, sin.Date_de_completude, sin.Date_de_decision) 
    - 24 * (
      (DATEDIFF(sin.Date_de_decision, sin.Date_de_completude) + 1) -- total days
      - (DATEDIFF(sin.Date_de_decision, sin.Date_de_completude) + 1 
         - SUM(CASE WHEN WEEKDAY(DATE_ADD(sin.Date_de_completude, INTERVAL seq DAY)) IN (5,6) THEN 1 ELSE 0 END)
      )
    )
  ) AS delai_decision_hour_skip_weekend
-- No join needed, just use your date variables



SELECT 
  SUM(
    CASE 
      WHEN WEEKDAY(DATE_ADD(sin.Date_de_completude, INTERVAL seq HOUR)) BETWEEN 0 AND 4 -- Monday to Friday
      AND HOUR(DATE_ADD(sin.Date_de_completude, INTERVAL seq HOUR)) BETWEEN 8 AND 16 -- 8 AM to 4:59 PM
      THEN 1
      ELSE 0
    END
  ) AS working_hours_excluding_weekends
-- No joins needed, just use your date variables


WITH RECURSIVE seq AS (
  SELECT 0 AS n
  UNION ALL
  SELECT n + 1
  FROM seq
  WHERE n < TIMESTAMPDIFF(HOUR, sin.Date_de_completude, sin.Date_de_decision)
)
SELECT
  SUM(
    CASE
      WHEN WEEKDAY(DATE_ADD(sin.Date_de_completude, INTERVAL seq.n HOUR)) BETWEEN 0 AND 4
       AND HOUR(DATE_ADD(sin.Date_de_completude, INTERVAL seq.n HOUR)) BETWEEN 8 AND 16
      THEN 1 ELSE 0
    END
  ) AS working_hours_excluding_weekends
FROM seq;