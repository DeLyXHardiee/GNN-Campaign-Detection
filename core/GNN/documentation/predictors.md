# Edge scorers: `DotPredictor` vs `MLPredictor`

You use these heads to turn **pairs of node embeddings** into a **score (logit)** for link prediction.

## Inputs they expect

* `src`: embedding matrix of the **source endpoints** of dimension d of the B edges to score → shape `[B, d]`
* `dst`: embedding matrix of the **destination endpoints** → shape `[B, d]`
* Output: logits → shape `[B]` (apply `sigmoid` at eval time if you want probabilities)

## 1) `DotPredictor`
* **What it does:** scores an edge by **similarity** of the two embeddings (inner product).
* **If you L2-normalize** embeddings first, the dot product becomes **cosine similarity**.
* **Params:** none (no learnable weights).
* **Pros:** tiny, fast, hard to overfit.
  **Cons:** fixed notion of “connected = similar”; can’t learn nonlinear patterns.

---

## 2) `MLPredictor`

* **What it does:** concatenates the two embeddings `[src; dst] ∈ R^{2d}` and runs a small **MLP** to output a logit.
* **Learns** a decision boundary from data (nonlinear + **order-aware**, which matches **directed** relations).
* **Params:** yes (two linear layers).
* **Pros:** more flexible; can learn “connected” patterns beyond simple similarity.
  **Cons:** more parameters → easier to overfit if data is small.

**Tip (undirected edges):** make it order-invariant by feeding symmetric features, e.g. `torch.cat([src*dst, (src-dst).abs()], dim=-1)`.

With `DotPredictor`, only the **GNN** has learnable params; with `MLPredictor`, **both** the GNN **and** the MLP head learn jointly.
