
1. **email** — one per spam email
    
    - **Attributes**
        
        - `x_text`: dense text embedding of `subject + body` (e.g., 384-dim MiniLM)
            
        - `ts`: UNIX timestamp (int64)
            
        - `n_urls`: count of unique URL domains in the body (int16)
            
        - `len_subject`, `len_body`: character counts (int32)
            
2. **url_domain** — eTLD+1 extracted from links in the body (your main campaign “glue”)
    
    - **Attributes**
		* `x_lex`: small lexical feature vector (length, #digits, #hyphens, char entropy) (e.g., 8–12 dims)
            
        - `docfreq`: #emails that used this domain (int32)
            
        - `first_seen`, `last_seen`: timestamps if you compute them (int64)
            
3. **url_stem** — **domain + first path segment** (e.g., `brand.co.uk/login`)
    
    - **Attributes**
        
        - Same as `url_domain` 
                
4. **sender_domain** — eTLD+1 of the sender address
    
    - **Attributes**
        
        - `x_lex`: same style as `url_domain` (length, digits, hyphens, entropy)
            
        - `docfreq`: #emails from this domain (int32)
    
    - is_freemail is used to avoid creating nodes for freemails.

5. **receiver_domain** - eTLD+1 of the receiver address
	* **Attributes**
		* same as sender_domain
            

## Edge types

- `("email", "contains", "url_domain")`
    
- `("email", "contains_stem", "url_stem")`
    
- `("sender_domain", "sent", "email")`

- `("receiver_domain", "received", "email")`

- **Reverse edges** for all of the above (use `ToUndirected()` in PyG once when building)
    

### Edge attributes (nice-to-have)

- `weight`: e.g., TF-IDF or 1.0 (float32)
    
- `ts`: copy the email timestamp onto the `email→artifact` edges (int64)
    

---

# Optional (add later if you ingest MISP or extra fields)

## Extra node types

- **attachment_hash** (sha256/md5 of attachments)
    
    - attrs: none or rarity (`docfreq`)
        
- **ip** (resolved IP of landing hosts)
    
    - attrs: ASN id, /24 prefix id (small embeddings), geo (optional)
        
- **hostname** (full FQDN if you want domain↔hostname hierarchy)
    
    - attrs: subdomain depth, lexical features
        
- **header_artifact** (e.g., `message_id`, `x_mailer`, `return_path`)
    
    - attrs: hashed embedding
        
- **campaign/actor** (MISP galaxy clusters) — for **evaluation** or weak supervision
    
    - attrs: one-hot id or small learned embedding
        
- **time_bucket** (e.g., day/week buckets) — if you prefer discrete temporal nodes
    
    - attrs: bucket index only
        

## Extra edge types

- `("email","has_attachment","attachment_hash")`
    
- `("url_domain","resolves_to","ip")`
    
- `("url_stem","in_domain","url_domain")` _(1–N mapping; helps share signal across stems of same domain)_
    
- `("email","header","header_artifact")`
    
- `("artifact","tagged_with","campaign")` (where _artifact_ ∈ {url_domain, url_stem, attachment_hash, ip})
    
- `("email","in_bucket","time_bucket")`
    

---
