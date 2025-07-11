
SELECT DATE(uploaded_date_time) AS date,
COUNT(*)/COUNT(DISTINCT uploaded_by) AS avg_docs_uploaded_peruser
FROM file_upload
GROUP BY DATE(uploaded_date_time)
ORDER BY date;


