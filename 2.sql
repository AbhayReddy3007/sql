SELECT DATE(ai_engine_call_datetime) AS date,
COUNT(DISTINCT user_id) AS active_users
FROM ai_engine_calls
GROUP BY date
ORDER BY date;

SELECT AVG(active_users) AS avg_active FROM( SELECT
COUNT(DISTINCT user_id) AS active_users,DATE(ai_engine_call_datetime)
FROM ai_engine_calls
GROUP BY DATE(ai_engine_call_datetime)
ORDER BY DATE(ai_engine_call_datetime)) AS daily_active
