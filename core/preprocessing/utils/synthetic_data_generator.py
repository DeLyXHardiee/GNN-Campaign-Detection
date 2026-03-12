# Generate a synthetic dataset of 50 emails following the user's schema and save as JSON

import json, random, hashlib, time
from email.utils import formatdate

categories = ["legitimate", "marketing", "spam", "phishing"]
domains = ["example.com","store.com","mailservice.net","secure-bank.com","company.dk"]
providers = ["gmail.com","yahoo.com","outlook.com","proton.me"]
colors = ["#000000","#1a73e8","#ff6600","#0078d4","#2a9d8f"]

def rand_hash():
    return hashlib.sha256(str(random.random()).encode()).hexdigest()

def to_str(v):
    if isinstance(v, str):
        return v
    return json.dumps(v)

def rfc_timestamp():
    return formatdate(usegmt=True)

events = []

base_epoch = 1773100000

for i in range(1, 51):
    cat = random.choices(categories, weights=[0.55,0.25,0.15,0.05])[0]
    domain = random.choice(domains)
    to_domain = random.choice(providers)
    from_email = f"sender{i}@{domain}"
    to_email = f"user{i}@{to_domain}"
    
    links = random.randint(0,3)
    images = random.randint(0,2)
    
    attachments = []
    if random.random() < 0.25:
        attachments = [rand_hash() for _ in range(random.randint(1,2))]
    
    auth = {
        "legitimate": "spf=pass; dkim=pass; dmarc=pass; header.from="+domain,
        "marketing": "spf=pass; dkim=pass; dmarc=pass; header.from="+domain,
        "spam": "spf=fail; dkim=none; dmarc=fail; header.from="+domain,
        "phishing": "spf=fail; dkim=none; dmarc=fail; header.from="+domain
    }[cat]
    
    scl = {"legitimate":-1,"marketing":1,"spam":8,"phishing":7}[cat]
    
    event = {
        "Event": {
            "info": "Synthetic dataset email",
            "email_index": i,
            "external_id": f"evt_{i:04d}",
            "Attribute": [
                {"type":"from","value":[from_email]},
                {"type":"to","value":[to_email]},
                {"type":"subject","value":f"Sample email subject {i}"},
                {"type":"date","value":to_str(base_epoch + i)},
                {"type":"body","value":"This is synthetic email content generated for dataset testing."},
                
                {"type":"html","value":{
                    "tag_counts":{"html":1,"head":1,"meta":1,"style":random.randint(0,1),"body":1,"table":random.randint(0,2),
                                  "tbody":random.randint(0,2),"tr":random.randint(0,4),"td":random.randint(0,8),"div":random.randint(0,3),
                                  "a":links,"p":random.randint(1,3),"o:p":0,"u":0,"ul":random.randint(0,1),"li":random.randint(0,3),
                                  "b":random.randint(0,2),"span":random.randint(0,2),"br":random.randint(0,3),"img":images},
                    "tree_stats":{"total_elements":random.randint(8,50),"max_depth":random.randint(3,7),
                                  "avg_depth":round(random.uniform(2.0,3.8),2),"forms":0,
                                  "password_fields":1 if cat=="phishing" and random.random()<0.5 else 0,
                                  "hidden_elements":random.randint(0,1),"external_scripts":0,
                                  "links":links,"images":images,
                                  "link_ratio":round(random.uniform(0.02,0.15),3),
                                  "image_ratio":round(random.uniform(0.0,0.08),3)},
                    "structure_fingerprint": hashlib.md5(f"fp{i}".encode()).hexdigest()
                }},
                
                {"type":"css","value":{"style_features":{
                    "unique_color_count":random.randint(1,5),
                    "primary_color":random.choice(colors),
                    "uses_position_absolute":False,
                    "uses_z_index":False,
                    "uses_media_queries":random.random()<0.4,
                    "unique_class_count":random.randint(0,6),
                    "class_entropy":round(random.uniform(0.0,2.5),2)
                }}},
                
                {"type":"attachments","value":attachments},
                {"type":"url","value":[f"https://link{i}-{j}.{domain}/page" for j in range(links)]},
                {"type":"category","value":cat},
                {"type":"rfc_defects","value":[]},
                {"type":"cyrillic_domain","value":to_str(False)},
                {"type":"contains_symbols","value":to_str(random.random()<0.3)},
                {"type":"body_has_tracking_url","value":to_str(links>0)},
                {"type":"body_has_tracking_image","value":to_str(images>0)},
                {"type":"body_has_tracking_pixel","value":to_str(images>0 and random.random()<0.5)},
                {"type":"body_has_unsubscribe_link","value":to_str(cat=="marketing")},
                {"type":"domain_is_common_webprovided","value":to_str(False)},
                
                {"type":"header_Received","value":[
                    {"origin_ip":"203.0.113."+str(random.randint(1,200)),"helo_host":"mail."+domain,
                     "by_host":"mx."+domain,"timestamp": rfc_timestamp()},
                    {"origin_ip":"203.0.113."+str(random.randint(1,200)),"helo_host":"mx."+domain,
                     "by_host":"mailbox."+to_domain,"timestamp": rfc_timestamp()}
                ]},
                
                {"type":"header_Return-Path","value":{"email":from_email,"domain":domain}},
                {"type":"header_Content-Type","value":["multipart/alternative","text/html"]},
                {"type":"header_Received-SPF","value":f"domain={domain}; helo=mail.{domain}"},
                {"type":"header_List-Unsubscribe","value":"<https://unsubscribe."+domain+">" if cat=="marketing" else ""},
                {"type":"header_Authentication-Results","value":auth},
                {"type":"header_X-Forefront-Antispam-Report","value":f"CTRY:US; LANG:en; SCL:{scl}; SFV:NSPM; CAT:{cat.upper()}"},
                {"type":"header_X-MS-Exchange-Organization-SCL","value":[to_str(scl)]}
            ]
        }
    }
    
    events.append(event)

dataset = {"Events": events}

path = "../../../data/misp/synthetic_email_dataset_50.json"
with open(path, "w") as f:
    json.dump(dataset, f, indent=2)

path