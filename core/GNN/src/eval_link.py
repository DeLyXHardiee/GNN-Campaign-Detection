# --- Collect logits & labels on TEST for AUROC/AP ---
import torch
from .embed import embed_with_graph
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def collect_scores(DEVICE, model, predictor, loaders_test):
    model.eval(); predictor.eval()
    all_scores = {}
    with torch.no_grad():
        for et, loader in loaders_test.items():
            ys, yhats = [], []
            for batch in loader:
                batch = batch.to(DEVICE)
                h = model(batch.x_dict, batch.edge_index_dict)
                idx = batch[et].edge_label_index
                y = batch[et].edge_label.float().cpu().numpy()
                src_type,_,dst_type = et
                s = h[src_type][idx[0]]; d = h[dst_type][idx[1]]
                logits = predictor(s,d).cpu().numpy()
                ys.append(y); yhats.append(logits)
            y = np.concatenate(ys); z = np.concatenate(yhats)
            auroc = roc_auc_score(y, z)
            ap    = average_precision_score(y, z)
            all_scores[et] = {'auroc': auroc, 'ap': ap}
    return all_scores

import torch
import math

# ---------- 1) Helpers ----------

import torch
import torch.nn.functional as F

def recall_at_k_mrr(h, edge_type, test_edges, K=20, use_dot=True, restrict_to_sources_with_pos=True):
    """
    h: dict {node_type: embeddings [N_t, d]} (all on SAME device)
    edge_type: (src_type, rel, dst_type)
    test_edges: edge_index [2, E_test] (same device as embeddings OR will be moved)
    """
    src_t, _, dst_t = edge_type
    S = h[src_t]
    D = h[dst_t]

    device = S.device
    test_edges = test_edges.to(device)

    if not use_dot:  # cosine
        S = F.normalize(S, p=2, dim=1)
        D = F.normalize(D, p=2, dim=1)

    Ns, Nd = S.size(0), D.size(0)

    # ground-truth adjacency for evaluation
    gt = torch.zeros((Ns, Nd), dtype=torch.bool, device=device)
    gt[test_edges[0], test_edges[1]] = True

    # restrict to sources that actually have at least one true edge in test
    if restrict_to_sources_with_pos:
        src_mask = gt.any(dim=1)
    else:
        src_mask = torch.ones(Ns, dtype=torch.bool, device=device)

    S_eval = S[src_mask]
    gt_eval = gt[src_mask]

    # compute scores and top-K
    scores = S_eval @ D.T                         # [Ns_eval, Nd]
    K = min(K, Nd)
    topk = torch.topk(scores, K, dim=1).indices   # [Ns_eval, K]

    # check which of top-K are true edges
    hits = gt_eval.gather(1, topk)                # same device now

    recall_k = hits.any(dim=1).float().mean().item()

    has = hits.any(dim=1)
    first_pos = torch.argmax(hits.int(), dim=1)
    ranks = torch.full((hits.size(0),), float('inf'), device=device)
    ranks[has] = (first_pos[has] + 1).float()
    mask = ranks != float('inf')
    mrr = (1.0 / ranks[mask]).mean().item() if mask.any() else 0.0

    return {
        'recall@K': recall_k,
        'MRR': mrr,
        'K': K,
        'n_eval_sources': int(hits.size(0)),
    }

def topk_eval_with_splits(DEVICE, model, splits, edge_types, K=20, use_dot=True):
    h_train = embed_with_graph(DEVICE, model, splits['train_graph'])   # leakage-safe embeddings
    results = {}
    for et in edge_types:
        res = recall_at_k_mrr(h_train, et, splits['test_pos'][et], K=K, use_dot=use_dot)
        results[et] = res
    return results

def topk_for_source(h, et, src_id, K=20, cosine=True):
    src_t,_,dst_t = et
    S = h[src_t]; D = h[dst_t]
    if cosine:
        S = torch.nn.functional.normalize(S, p=2, dim=1)
        D = torch.nn.functional.normalize(D, p=2, dim=1)
    s = S[src_id:src_id+1]
    scores = (s @ D.T).squeeze(0).cpu()
    vals, idxs = torch.topk(scores, min(K, D.size(0)))
    return idxs.tolist(), vals.tolist()

