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

def recall_at_k_mrr(h, edge_type, test_edges, K=20, use_dot=True, restrict_to_sources_with_pos=True):
    src_t, _, dst_t = edge_type
    S = h[src_t]; D = h[dst_t]
    if not use_dot:  # cosine
        S = torch.nn.functional.normalize(S, p=2, dim=1)
        D = torch.nn.functional.normalize(D, p=2, dim=1)

    Ns, Nd = S.size(0), D.size(0)
    gt = torch.zeros((Ns, Nd), dtype=torch.bool)
    gt[test_edges[0], test_edges[1]] = True

    src_mask = gt.any(dim=1) if restrict_to_sources_with_pos else torch.ones(Ns, dtype=torch.bool)
    S_eval = S[src_mask]; gt_eval = gt[src_mask]

    scores = S_eval @ D.T
    K = min(K, Nd)
    topk = torch.topk(scores, K, dim=1).indices
    hits = gt_eval.gather(1, topk)

    recall_k = hits.any(dim=1).float().mean().item()
    has = hits.any(dim=1)
    first_pos = torch.argmax(hits.int(), dim=1)
    ranks = torch.full((hits.size(0),), float('inf'))
    ranks[has] = (first_pos[has] + 1).float()
    mrr = (1.0 / ranks[ranks != float('inf')]).mean().item() if (ranks != float('inf')).any() else 0.0
    return {'recall@K': recall_k, 'MRR': mrr, 'K': K, 'n_eval_sources': int(hits.size(0))}

def topk_eval_with_splits(model, splits, edge_types, K=20, use_dot=True):
    h_train = embed_with_graph(model, splits['train_graph'])   # leakage-safe embeddings
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

