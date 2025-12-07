import torch
from .model import HeteroSAGE, DotPredictor, MLPredictor
from .loaders import make_link_loaders
from .build_graph_splits import pick_supervised_edge_types, split_edges_and_build_train_graph
from .model_io import save_model_checkpoint, load_training_state
from torch import nn

# ---------- Step 7: Train / Eval ----------
# train.py (or wherever batch_loss lives)
import torch
import torch.nn.functional as F

def batch_loss(model, predictor, batch, edge_type, pos_weight_fixed=None):
    h_dict = model(batch.x_dict, batch.edge_index_dict)
    e_store = batch[edge_type]
    idx = e_store.edge_label_index            # [2, B_total]
    y   = e_store.edge_label.float()          # [B_total] in {0,1}

    src_t, _, dst_t = edge_type
    src = h_dict[src_t][idx[0]]
    dst = h_dict[dst_t][idx[1]]
    logits = predictor(src, dst)

    # ---- Stable pos_weight ----
    if isinstance(pos_weight_fixed, dict):
        # per-relation weight (optional enhancement)
        pw = torch.tensor(float(pos_weight_fixed.get(edge_type, 1.0)), device=logits.device)
    elif isinstance(pos_weight_fixed, (int, float)):
        pw = torch.tensor(float(pos_weight_fixed), device=logits.device)

    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

    with torch.no_grad():
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()
        acc  = (pred == y).float().mean().item()

    return loss, acc


def train_epoch(DEVICE, model, predictor, optimizer, loaders_train, pos_weight_fixed=1.0):
    model.train(); predictor.train()
    total_loss, total_acc, total_batches = 0.0, 0.0, 0
    for et, loader in loaders_train.items():
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss, acc = batch_loss(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); total_acc += acc; total_batches += 1
    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)

@torch.no_grad()
def eval_epoch(DEVICE, model, predictor, loaders_eval, pos_weight_fixed=1.0):
    model.eval(); predictor.eval()
    total_loss, total_acc, total_batches = 0.0, 0.0, 0
    for et, loader in loaders_eval.items():
        for batch in loader:
            batch = batch.to(DEVICE)
            loss, acc = batch_loss(model, predictor, batch, et, pos_weight_fixed=pos_weight_fixed)
            total_loss += loss.item(); total_acc += acc; total_batches += 1
    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)

def run_training(DEVICE, TORCH_SEED, data,
                 primary_ntype='movie',
                 hidden=128, out_dim=128, layers=2, dropout=0.1,
                 neg_ratio=1.0, batch_size=1024, fanout=[15, 10],
                 val_ratio=0.1, test_ratio=0.1, epochs=5, lr=1e-3, wd=1e-4,
                 score_head='dot', early_stopping_patience=5,
                 model_save_name="best_model.pt",
                 resume_from=None):   # 'dot' or 'mlp'
    # Force loaders/sampling to run on CPU to avoid MPS CSR limitations; batches are moved to DEVICE later.
    data_cpu = data.to('cpu')
    print("Metadata:", data_cpu.metadata())

    # Training configuration for checkpointing/resume
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
    }

    if resume_from:
        model, predictor, opt, loaders, splits, checkpoint, start_epoch, patience_counter, best_val, best_state = load_training_state(
            data=data_cpu,
            device=DEVICE,
            filename=resume_from,
        )
        loader_params = checkpoint.get('loader_params', loader_params)
        training_params = checkpoint.get('training_params', training_params)
        train_graph = splits['train_graph']
        train_pos = splits['train_pos']
        val_pos = splits['val_pos']
        test_pos = splits['test_pos']
        sup_ets = splits['sup_ets']
        pos_weight_fixed = float(loader_params['neg_ratio'])
        print(f"Resuming from epoch {start_epoch} (best val {best_val:.4f}) using checkpoint '{resume_from}'")
    else:
        sup_ets = pick_supervised_edge_types(data, primary_ntype=primary_ntype, direction='both')
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

        if score_head == 'mlp':
            predictor = MLPredictor(out_dim).to(DEVICE)
        else:
            predictor = DotPredictor().to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        pos_weight_fixed = float(neg_ratio)
        best_val = float('inf'); best_state = None; patience_counter = 0; start_epoch = 0

    print("Starting training!")
    for epoch in range(start_epoch + 1, epochs + 1):
        tr_loss, tr_acc = train_epoch(DEVICE, model, predictor, opt, loaders['train'], pos_weight_fixed=pos_weight_fixed)
        va_loss, va_acc = eval_epoch(DEVICE, model, predictor, loaders['val'], pos_weight_fixed=pos_weight_fixed)
        print(f"[Epoch {epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            patience_counter = 0
            best_state = {'model': model.state_dict(), 'pred': predictor.state_dict()}
            save_path = save_model_checkpoint(
                model=model,
                predictor=predictor,
                sup_edge_types=sup_ets,
                epoch=epoch,
                val_loss=va_loss,
                config=checkpoint_config,
                data_metadata=data_cpu.metadata(),
                train_pos=train_pos,
                val_pos=val_pos,
                test_pos=test_pos,
                loader_params=loader_params,
                torch_seed=TORCH_SEED,
                optimizer_state=opt.state_dict(),
                patience_counter=patience_counter,
                best_val=best_val,
                best_model_state=best_state['model'],
                best_predictor_state=best_state['pred'],
                training_params=training_params,
                filename=model_save_name,
                )
            print(f"✓ Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs without val loss improvement.")
                break

    if best_state:
        model.load_state_dict(best_state['model'])
        predictor.load_state_dict(best_state['pred'])
    te_loss, te_acc = eval_epoch(DEVICE, model, predictor, loaders['test'])
    print(f"[Test] loss {te_loss:.4f} acc {te_acc:.3f}")

    # Return everything you’ll want for evaluation
    splits = {
        'train_graph': train_graph,
        'train_pos': train_pos,
        'val_pos': val_pos,
        'test_pos': test_pos,
        'sup_ets': sup_ets,
    }
    return model, predictor, loaders, splits
