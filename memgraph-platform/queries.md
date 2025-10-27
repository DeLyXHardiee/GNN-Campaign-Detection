Count number of stems connected to more than one email (include count of total stems)

MATCH (s:Stem)<-[:HAS_STEM]-(:Url)<-[:HAS_URL]-(e:Email)
WITH s, count(DISTINCT e) AS email_count
RETURN
  sum(CASE WHEN email_count > 1 THEN 1 ELSE 0 END) AS stems_with_multiple_emails,
  count(s) AS total_stems;


Count stems with multiple emails and those sharing a domain

MATCH (s:Stem)<-[:HAS_STEM]-(u:Url)<-[:HAS_URL]-(e:Email)
MATCH (u)-[:HAS_DOMAIN]->(d:Domain)
WITH s, d, count(DISTINCT e) AS emails_per_domain, count(DISTINCT e) AS total_emails
WITH s, sum(emails_per_domain) AS sum_emails, 
         max(emails_per_domain) AS max_emails_per_domain
WHERE sum_emails > 1
RETURN
    count(s) AS stems_with_multiple_emails,
    count(CASE WHEN max_emails_per_domain > 1 THEN 1 END) AS stems_sharing_domain;


