SELECT(
(SELECT COUNT(DISTINCT ai_engine_calls.user_id) AS used_prod FROM ai_engine_calls)*1.00/
(SELECT COUNT(DISTINCT application_users.id) AS user_with_access FROM application_users))*100 AS used_once_percentage


