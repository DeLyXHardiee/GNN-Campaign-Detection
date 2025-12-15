import torch
from .model import HeteroSAGE, DotPredictor, MLPredictor, DistMultPredictor
from .loaders import make_link_loaders
from .build_graph_splits import pick_supervised_edge_types, split_edges_and_build_train_graph
from .model_io import save_model_checkpoint, load_training_state
from torch import nn
import time

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torch.nn.functional as F

def batch_loss(model, predictor, batch, edge_type, pos_weight_fixed=None):
    h_dict = model(batch.x_dict, batch.edge_index_dict)
    e_store = batch[edge_type]
    idx = e_store.edge_label_index
    y = e_store.edge_label.float()

    src_t, _, dst_t = edge_type
    src = h_dict[src_t][idx[0]]
    dst = h_dict[dst_t][idx[1]]
    logits = predictor(src, dst, edge_type)

    if isinstance(pos_weight_fixed, dict):
        pw = torch.tensor(float(pos_weight_fixed.get(edge_type, 1.0)), device=logits.device)
    elif isinstance(pos_weight_fixed, (int, float)):
        pw = torch.tensor(float(pos_weight_fixed), device=logits.device)

    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        acc = (pred == y).float().mean().item()

    return loss, acc

def batch_loss_contrastive(model, predictor, batch, edge_type,
                           pos_weight_fixed=None,
                           contrastive_edges=None,
                           contrastive_weight=0.2):
    """
    BCE link-prediction loss + GraphStorm-style contrastive term.

    - BCE uses logits vs y (0/1) with optional pos_weight.
    - Contrastive term (if enabled for this edge_type) is:

        loss_contr = - mean_i log(
            exp(pos_score_i) / sum_j exp(score_{i,j})
        )

      where for each positive edge i, the denominator sums over
      {that positive} U {all negatives in the batch}.
    """
    import torch
    import torch.nn.functional as F

    # Forward pass to get embeddings and logits
    h_dict = model(batch.x_dict, batch.edge_index_dict)
    e_store = batch[edge_type]
    idx = e_store.edge_label_index
    y   = e_store.edge_label.float()     # [B] in {0,1}

    src_t, _, dst_t = edge_type
    src = h_dict[src_t][idx[0]]
    dst = h_dict[dst_t][idx[1]]
    logits = predictor(src, dst, edge_type)

    if isinstance(pos_weight_fixed, dict):
        pw = torch.tensor(float(pos_weight_fixed.get(edge_type, 1.0)), device=logits.device)
    elif isinstance(pos_weight_fixed, (int, float)):
        pw = torch.tensor(float(pos_weight_fixed), device=logits.device)
    else:
        pw = torch.tensor(1.0, device=logits.device)

    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

    use_contrastive = (contrastive_edges is None) or (edge_type in contrastive_edges)

    if use_contrastive and contrastive_weight > 0.0:
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]

        if pos_scores.numel() > 0 and neg_scores.numel() > 0:
            P = pos_scores.shape[0]
            N = neg_scores.shape[0]

            pos_expanded = pos_scores.view(P, 1)                
            neg_expanded = neg_scores.view(1, N).expand(P, N)
            all_scores = torch.cat([pos_expanded, neg_expanded], dim=1)

            log_denom = torch.logsumexp(all_scores, dim=1)

            contrastive_loss = -(pos_scores - log_denom).mean()

            loss = loss + contrastive_weight * contrastive_loss

    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        acc  = (pred == y).float().mean().item()

    return loss, acc


def train_epoch(DEVICE, model, predictor, optimizer, loaders_train, pos_weight_fixed=1.0, 
                contrastive_edges=None, contrastive_weight=0.2):
    model.train(); predictor.train()
    total_loss, total_acc, total_batches = 0.0, 0.0, 0

    total_steps = sum(len(loader) for loader in loaders_train.values())
    pbar = tqdm(total=total_steps, desc="🪠 Training", leave=True)

    for et, loader in loaders_train.items():
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            if contrastive_edges:
                loss, acc = batch_loss_contrastive(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed,
                                                   contrastive_edges=contrastive_edges,
                                                   contrastive_weight=contrastive_weight)
            else:
                loss, acc = batch_loss(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            total_batches += 1
            pbar.update(1)

    pbar.close()
    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)

@torch.no_grad()
def eval_epoch(DEVICE, model, predictor, loaders_eval, pos_weight_fixed=1.0, 
               contrastive_edges=None, contrastive_weight=0.2):
    model.eval(); predictor.eval()
    total_loss, total_acc, total_batches = 0.0, 0.0, 0

    total_steps = sum(len(loader) for loader in loaders_eval.values())
    pbar = tqdm(total=total_steps, desc="🧚 Evaluating", leave=True)

    for et, loader in loaders_eval.items():
        for batch in loader:
            batch = batch.to(DEVICE)
            if contrastive_edges:
                loss, acc = batch_loss_contrastive(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed,
                                                   contrastive_edges=contrastive_edges,
                                                   contrastive_weight=contrastive_weight)
            else:
                loss, acc = batch_loss(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed)
            total_loss += loss.item()
            total_acc += acc
            total_batches += 1
            pbar.update(1)

    pbar.close()
    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)



def run_training(DEVICE, TORCH_SEED, data,
                 primary_ntype='movie',
                 hidden=128, out_dim=128, layers=2, dropout=0.1,
                 neg_ratio=1.0, batch_size=1024, fanout=[15, 10],
                 val_ratio=0.1, test_ratio=0.1, epochs=5, lr=1e-3, wd=1e-4,
                 score_head='mlp', early_stopping_patience=5,
                 lr_reduce_patience=5, lr_reduce_factor=0.5, lr_reduce_min=0.0,
                 supervised_edge_types=None,
                 model_save_name="best_model.pt",
                 contrastive_edges=None,
                 contrastive_weight=0.2):
    import os
    import csv
    import json
    from datetime import datetime

    from pathlib import Path

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("run-%Y-%m-%d_%H-%M-%S")
    run_dir = models_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔖 Saving run artifacts to: {run_dir}")

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "primary_ntype": primary_ntype,
            "hidden": hidden,
            "out_dim": out_dim,
            "layers": layers,
            "dropout": dropout,
            "neg_ratio": neg_ratio,
            "batch_size": batch_size,
            "fanout": fanout,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": wd,
            "score_head": score_head,
            "contrastive_edges": contrastive_edges,
            "supervised_edges": supervised_edge_types,
            "contrastive_weight": contrastive_weight
        }, f, indent=2)

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_loss', 'val_acc'])

    data_cpu = data.to('cpu')
    print("Metadata:", data_cpu.metadata())

    checkpoint_config = {
        'hidden': hidden,
        'out_dim': out_dim,
        'layers': layers,
        'dropout': dropout,
        'score_head': score_head,
    }
    loader_params = {
        'neg_ratio': neg_ratio,
        'batch_size': batch_size,
        'fanout': fanout,
    }
    training_params = {
        'lr': lr,
        'wd': wd,
        'target_epochs': epochs,
        'lr_reduce_patience': lr_reduce_patience,
        'lr_reduce_factor': lr_reduce_factor,
        'lr_reduce_min': lr_reduce_min,
    }

    sup_ets = pick_supervised_edge_types(
        data,
        primary_ntype=primary_ntype,
        direction='both',
        supervised_edge_types=supervised_edge_types,
    )
    print("Supervised edge types:", sup_ets)

    train_graph, train_pos, val_pos, test_pos = split_edges_and_build_train_graph(TORCH_SEED,
        data, sup_ets, val_ratio=val_ratio, test_ratio=test_ratio
    )
    print("Build train graph!")

    loaders = make_link_loaders(
        train_graph=train_graph, full_graph=data,
        train_pos=train_pos, val_pos=val_pos, test_pos=test_pos,
        edge_types=sup_ets, neg_ratio=neg_ratio, batch_size=batch_size, fanout=fanout
    )

    print("Build link loaders!")

    model = HeteroSAGE(metadata=data.metadata(),
                       hidden=hidden, out=out_dim, layers=layers, dropout=dropout).to(DEVICE)

    if score_head == "mlp":
        predictor = MLPredictor(out_dim).to(DEVICE)
    elif score_head == "distmult":
        predictor = DistMultPredictor(out_dim, sup_ets).to(DEVICE)
    else:
        predictor = DotPredictor().to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=lr_reduce_min,
    )

    pos_weight_fixed = float(neg_ratio)
    best_val = float('inf'); best_state = None
    patience_counter = 0; start_epoch = 0

    print("Starting training!")
    for epoch in range(start_epoch + 1, epochs + 1):
        tr_loss, tr_acc = train_epoch(DEVICE, model, predictor, opt, loaders['train'], pos_weight_fixed=pos_weight_fixed)
        va_loss, va_acc = eval_epoch(DEVICE, model, predictor, loaders['val'], pos_weight_fixed=pos_weight_fixed)
        print(f"🧪 Epoch {epoch:02d} | 🏋️ train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | 📉 val loss: {va_loss:.4f} acc: {va_acc:.3f}")

        with open(metrics_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, va_loss, va_acc])

        if epoch % 5 == 0 or epoch == 1:
            save_model_checkpoint(
                model=model, predictor=predictor, sup_edge_types=sup_ets,
                epoch=epoch, val_loss=va_loss, config=checkpoint_config,
                data_metadata=data_cpu.metadata(), train_pos=train_pos, val_pos=val_pos,
                test_pos=test_pos, loader_params=loader_params, torch_seed=TORCH_SEED,
                optimizer_state=opt.state_dict(), scheduler_state=scheduler.state_dict(),
                patience_counter=patience_counter, best_val=best_val,
                best_model_state=model.state_dict(), best_predictor_state=predictor.state_dict(),
                training_params=training_params,
                save_dir=run_dir,
                filename=f"model_epoch_{epoch}.pt"
            )

        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            best_state = {'model': model.state_dict(), 'pred': predictor.state_dict()}
            save_model_checkpoint(
                model=model, predictor=predictor, sup_edge_types=sup_ets,
                epoch=epoch, val_loss=va_loss, config=checkpoint_config,
                data_metadata=data_cpu.metadata(), train_pos=train_pos, val_pos=val_pos,
                test_pos=test_pos, loader_params=loader_params, torch_seed=TORCH_SEED,
                optimizer_state=opt.state_dict(), scheduler_state=scheduler.state_dict(),
                patience_counter=patience_counter, best_val=best_val,
                best_model_state=best_state['model'], best_predictor_state=best_state['pred'],
                training_params=training_params, 
                save_dir=run_dir,
                filename=f"best_model.pt"
            )
            print(f"Best-accuracy model saved to {os.path.join(run_dir, 'model_best_val.pt')}")
        else:
            patience_counter += 1

        prev_lr = opt.param_groups[0]['lr']
        scheduler.step(va_loss)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"🔻 Learning rate reduced from {prev_lr:.2e} to {new_lr:.2e} due to plateau.")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs without val loss improvement.")
            break

    if best_state:
        model.load_state_dict(best_state['model'])
        predictor.load_state_dict(best_state['pred'])
    te_loss, te_acc = eval_epoch(DEVICE, model, predictor, loaders['test'])
    print(f"[Test] loss {te_loss:.4f} acc {te_acc:.3f}")

    splits = {
        'train_graph': train_graph,
        'train_pos': train_pos,
        'val_pos': val_pos,
        'test_pos': test_pos,
        'sup_ets': sup_ets,
    }
    return model, predictor, loaders, splits




