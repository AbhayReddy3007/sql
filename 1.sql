WITH first_use AS
(SELECT user_id,MIN(DATE(ai_engine_call_datetime)) AS first_use_date FROM ai_engine_calls GROUP BY user_id),
recent_use AS 
(SELECT user_id,DATE(ai_engine_call_datetime) AS recent_use_date FROM ai_engine_calls
WHERE DATE(ai_engine_call_datetime) > CURRENT_DATE - INTERVAL '30 days')

SELECT COUNT(DISTINCT recent_use.user_id) AS retuned
FROM recent_use
JOIN first_use ON recent_use.user_id=first_use.user_id
WHERE recent_use.recent_use_date>first_use_date;
