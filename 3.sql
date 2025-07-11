WITH 
rownumbers AS (
SELECT id,user_id,ai_engine_call_datetime ,ROW_NUMBER() OVER (ORDER BY ai_engine_call_datetime) 
AS rn FROM ai_engine_calls),

joining AS (
SELECT rnum.id,rnum.user_id,rnum.ai_engine_call_datetime,prev.user_id AS prev_user_id,
CASE
WHEN prev.user_id IS NULL OR rnum.user_id != prev.user_id
THEN 1 ELSE 0
END AS new_session
FROM rownumbers rnum
LEFT JOIN rownumbers prev ON rnum.rn = prev.rn +1),

sessions AS(
SELECT *, SUM(new_session) OVER (ORDER BY ai_engine_call_datetime 
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_group FROM joining),

usefull AS(
SELECT user_id,session_group, COUNT(*) AS prompts_in_session FROM sessions
GROUP BY user_id,session_group HAVING COUNT(*)>1),

Average AS( SELECT user_id,AVG(prompts_in_session)AS avg_prompts_per_session
FROM usefull GROUP BY user_id)

SELECT * FROM Average
ORDER BY user_id