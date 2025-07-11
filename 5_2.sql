
SELECT DATE(uploaded_date_time) AS date,
COUNT(*) AS total_no_of_docs
FROM file_upload
GROUP BY DATE(uploaded_date_time)
ORDER BY date;


