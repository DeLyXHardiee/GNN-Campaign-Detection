# What `HeteroSAGE` is

Hetero-sage converts SAGEBackbone (a homogeneus model) to a heterogeneus model. We convert it by using the to_hetero-method in the constructor. It receives meta-data about what relations exist in the graph in the constructor. It also receives arguments that are passed to SAGEBackbone, which we have already described.

The forward method works like so:
* **x_dict**: a dictionary that contains **{node_type : x_t}**. 
    * **node_type**: could be "movie", "actor" or "director".
    * *x_t*: a list containing **[num_nodes_of_type_in_batch, feat_dim_for_that_type]**.
        * **num_nodes_of_type_in_batch**: the number of that node_type in the batch.
        * **feat_dim_for_that_type**: feature dimensions for the node_type.
* **edge_index_dict**: a dictionary that contains **{edge_type : edge_index_for_that_relation}**.
    * **edge_type**: the type of relation, for example movie -> actor.
    * **edge_index_for_that_relation**: a matrix of dimension [2, no_of_relations]. It works almost like a relational database that links nodes with an edge.

Behind the scenes, these are passed to the forward method of **SAGEBackbone**. It outputs a dictionary of node **embeddings**: **{node_type : z_t}**.
Each **z_t** is **[num_nodes_of_type_in_batch, out]**, where out are the embeddings.

**x_dict** and **edge_index_dict** are from a single batch of one of the NeighborLinkLoader's that we constructed in make_link_loaders.

Note about weights: Weight-matrices are indexed by (layer, relation). Each each layer has a two unique weight-matrix per. relation (one weight-matric for src and one weight-matrix for dst).