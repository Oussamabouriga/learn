## Calculating Working Hours (8:00–17:00) While Skipping Weekends

This SQL query calculates the total number of **working hours** between two dates:

- Only counts **Monday to Friday**  
- Only counts hours between **8:00 AM and 5:00 PM**  
- Skips **Saturday and Sunday**  
- Works hour-by-hour using a sequence table  

### SQL Query

```sql
SELECT 
  SUM(
    CASE 
      WHEN WEEKDAY(DATE_ADD(sin.Date_de_completude, INTERVAL seq HOUR)) BETWEEN 0 AND 4 
           -- Monday to Friday
      AND HOUR(DATE_ADD(sin.Date_de_completude, INTERVAL seq HOUR)) BETWEEN 8 AND 16 
           -- 8:00 to 16:59 (8 AM to 4:59 PM)
      THEN 1
      ELSE 0
    END
  ) AS working_hours_skip_weekend
FROM your_table sin
JOIN seq_0_to_N 
  ON seq <= TIMESTAMPDIFF(HOUR, sin.Date_de_completude, sin.Date_de_decision);


## Calculating Time Difference in Hours While Skipping Weekends

This SQL query calculates the **total time difference in hours** between two dates while:

- Skipping **Saturday and Sunday**
- Counting full days (24 hours per weekday)
- Returning the result in **whole hours**

### SQL Query

```sql
SELECT 
  FLOOR(
    TIMESTAMPDIFF(HOUR, sin.Date_de_completude, sin.Date_de_decision) 
    - 24 * (
      -- Count how many weekend days (Saturday=5, Sunday=6) are between the two dates
      (DATEDIFF(sin.Date_de_decision, sin.Date_de_completude) + 1)
      - (
          (DATEDIFF(sin.Date_de_decision, sin.Date_de_completude) + 1)
          - SUM(
              CASE 
                WHEN WEEKDAY(DATE_ADD(sin.Date_de_completude, INTERVAL seq DAY)) IN (5,6) 
                THEN 1 
                ELSE 0 
              END
            )
        )
    )
  ) AS delai_decision_hour_skip_weekend
FROM your_table sin
JOIN seq_0_to_N 
  ON seq <= DATEDIFF(sin.Date_de_decision, sin.Date_de_completude);