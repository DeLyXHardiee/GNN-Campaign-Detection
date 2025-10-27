### Visualize all nodes and edges for emails connected to week 14 in 2007

    MATCH (w:Week {key: "2007-W14"})<-[r_in:IN_WEEK]-(e:Email)
    OPTIONAL MATCH (e)-[r_sender:HAS_SENDER]->(s:Sender)-[r_sender_dom:FROM_DOMAIN]->(sd:EmailDomain)
    OPTIONAL MATCH (e)-[r_recv:HAS_RECEIVER]->(r:Receiver)-[r_recv_dom:FROM_DOMAIN]->(rd:EmailDomain)
    OPTIONAL MATCH (e)-[r_sub:HAS_SUBJECT]->(sub:Subject)
    OPTIONAL MATCH (e)-[r_url:HAS_URL]->(u:Url)-[r_dom:HAS_DOMAIN]->(d:Domain)
    OPTIONAL MATCH (u)-[r_stem:HAS_STEM]->(st:Stem)
    RETURN w, e, s, sd, r, rd, sub, u, d, st,
        r_in, r_sender, r_sender_dom, r_recv, r_recv_dom, 
        r_sub, r_url, r_dom, r_stem;

    
### Count stems connected to more than one email (include count of total stems)

    MATCH (s:Stem)<-[:HAS_STEM]-(:Url)<-[:HAS_URL]-(e:Email)
    WITH s, count(DISTINCT e) AS email_count
    RETURN
        sum(CASE WHEN email_count > 1 THEN 1 ELSE 0 END) AS stems_with_multiple_emails,
        count(s) AS total_stems;



### Count stems with multiple emails and those sharing a domain

    MATCH (s:Stem)<-[:HAS_STEM]-(u:Url)<-[:HAS_URL]-(e:Email)
    MATCH (u)-[:HAS_DOMAIN]->(d:Domain)
    WITH s, d, count(DISTINCT e) AS emails_per_domain, count(DISTINCT e) AS total_emails
    WITH s, sum(emails_per_domain) AS sum_emails, 
            max(emails_per_domain) AS max_emails_per_domain
    WHERE sum_emails > 1
    RETURN
        count(s) AS stems_with_multiple_emails,
        count(CASE WHEN max_emails_per_domain > 1 THEN 1 END) AS stems_sharing_domain;



### Count domains connected to more than one email (include count of total domains)

    MATCH (d:Domain)
    OPTIONAL MATCH (e:Email)-[:HAS_URL]->(:Url)-[:HAS_DOMAIN]->(d)
    WITH d, count(DISTINCT e) AS email_count
    RETURN
        count(CASE WHEN email_count > 1 THEN 1 END) AS domains_with_multiple_emails,
        count(d) AS total_domains;



### Count subjects connected to more than one email (include count of total subjects)

    MATCH (e:Email)-[:HAS_SUBJECT]->(s:Subject)
    WITH s, count(DISTINCT e) AS email_count
    RETURN
        count(CASE WHEN email_count > 1 THEN 1 END) AS subjects_with_multiple_emails,
        count(s) AS total_subjects;
