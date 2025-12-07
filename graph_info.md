# Graph Schema Information

This document describes the current heterogeneous graph schema used in the GNN Campaign Detection pipeline.

## Node Types

1. **email** — The central node representing a spam email.
    - **Features (x)**:
        - `ts`: UNIX timestamp (float).
        - `len_body`: Character count of the email body (float).
        - `n_urls`: Number of unique URLs in the email (float).
        - `len_subject`: Character count of the subject (float).
        - `x_text`: TF-IDF vector of `subject + body` (optional, configurable dims).
    - **Note**: In the PyTorch pipeline, these features are normalized (z-score/minmax) via `core/graph/normalizer.py`.

2. **sender** — The sender's email address.
    - **Features (x)**: `[length_of_address]`
    - **Attributes**: `docfreq` (number of emails with this sender).

3. **receiver** — The receiver's email address.
    - **Features (x)**: `[length_of_address]`
    - **Attributes**: `docfreq` (number of emails with this receiver).

4. **week** — Temporal bucket (ISO week string, e.g., "2007-W15").
    - **Features (x)**: `[index]` (sequential index of the week).

5. **url** — A full URL found in the email body.
    - **Features (x)**: `[path_length]` (length of the URL path/stem).
    - **Attributes**: `docfreq` (number of emails containing this URL).

6. **domain** — The eTLD+1 extracted from a URL.
    - **Features (x)**: `[entropy]` (character entropy of the domain string).
    - **Attributes**:
        - `x_lex`: Lexical feature vector (length, digits, hyphens, entropy, etc.).
        - `docfreq`: Number of emails containing URLs with this domain.

7. **stem** — The "stem" (path/query) extracted from a URL.
    - **Features (x)**: `[length]`
    - **Attributes**:
        - `x_lex`: Lexical feature vector.
        - `docfreq`: Number of emails containing URLs with this stem.

8. **email_domain** — The domain extracted from a sender or receiver email address.
    - **Features (x)**: `[length]`
    - **Attributes**:
        - `x_lex`: Lexical feature vector.
        - `docfreq_sender`: Count of sender occurrences.
        - `docfreq_receiver`: Count of receiver occurrences.

## Edge Types

The graph connects emails to their components and components to each other:

- **Direct Email Connections**:
    - `(email)-[has_sender]->(sender)`
    - `(email)-[has_receiver]->(receiver)`
    - `(email)-[in_week]->(week)`
    - `(email)-[has_url]->(url)`

- **Component Hierarchies**:
    - `(url)-[has_domain]->(domain)`
    - `(url)-[has_stem]->(stem)`
    - `(sender)-[from_domain]->(email_domain)`
    - `(receiver)-[from_domain]->(email_domain)`

## Implementation Details

- **Graph IR**: The graph is assembled into a backend-agnostic Intermediate Representation (IR) in `core/graph/assembler.py`.
- **PyTorch Geometric**: `core/graph/graph_builder_pytorch.py` converts the IR into a `HeteroData` object and applies normalization.
- **Memgraph**: `core/graph/graph_builder_memgraph.py` mirrors the IR into a Memgraph database via Bolt.

