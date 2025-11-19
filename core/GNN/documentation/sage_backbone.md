# What `SAGEBackbone` is

An **encoder**: it maps raw node features to fixed-size **node embeddings**. You’ll use those embeddings for link prediction, k-NN + clustering, etc.

# Constructor (what each arg means and what gets built)

**Args**

* **hidden**: width of the internal representation (number of **output channels** a non-final layer produces). E.g., 128 → each node has a 128-D vector after that layer.
* **out**: width of the **final** embedding (last layer’s output channels).
* **layers**: how many GraphSAGE layers to stack (each layer = one message-passing round).
* **dropout**: probability (e.g., 0.1 = 10%) of zeroing activations **between layers** during training to reduce overfitting.

**Modules**

* **`self.layers = nn.ModuleList()`**
  A registered container holding the stacked SAGE layers.
* **First layer: `SAGEConv((-1, -1), hidden)`**
  `(-1, -1)` = *infer source and destination input feature sizes at first use*.
  This makes the backbone work for **heterogeneous/bipartite** cases (src/dst types can have different input dims). Output has size `hidden`.
* **Middle layers (if any): `SAGEConv((hidden, hidden), hidden)`**
  After layer 1, all node representations are `hidden`-dim, so both inputs are known: `(hidden, hidden) → hidden`.
* **Final layer (if `layers > 1`): `SAGEConv((hidden, hidden), out)`**
  Maps the internal width to your desired **final embedding** size `out`.
* **`self.dropout = nn.Dropout(dropout)`**
  Applied after ReLU on all **non-final** layers (active only in `.train()`).

# Forward (what happens at inference/training time)

**Inputs**

* **`x`**: node-feature matrix shaped `[num_nodes, in_features]` (or per-type dict when wrapped with `to_hetero`).
* **`edge_index`**: a **[2, E]** tensor. **Each column** is one directed edge `(src, dst)`.
  (Small correction: it’s 2 **rows**, E columns—not two columns.)

**Loop**
For each SAGE layer:

1. **`h = conv(h, edge_index)`** → one message-passing round:

   * For every graph node, gather its **incoming neighbors’** vectors from the previous layer, take their **mean**,
   * Linearly mix **self** and **neighbor-mean** with learnable weights,
   * (PyG handles this efficiently; you don’t write the loops.)
2. If this is **not** the last layer: apply **ReLU** and **Dropout**.

Return the final `h` (shape `[num_nodes, out]` if `layers > 1`; otherwise `[num_nodes, hidden]`).

# Gotchas (easy to mix up)

* **“Channels” = features = embedding dimensions.**
* **Dropout** runs **between** layers only (not after the last).
* If **`layers == 1`**, there’s only the first layer; the output size is `hidden` and `out` is effectively ignored for this class.
* `edge_index` is **[2, E]**, each **column** an edge `(src, dst)`.
