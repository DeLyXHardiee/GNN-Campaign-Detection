import torch
from pathlib import Path

from .model import HeteroSAGE, DotPredictor, MLPredictor
from .loaders import make_link_loaders


def get_models_dir() -> Path:
    """
    Return the directory where model checkpoints are stored.
    Creates the directory if it does not exist.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def select_device(preferred=None):
    """
    Auto-pick a device (cuda > mps > cpu) unless a preferred one is provided.
    """
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model_checkpoint(
    model,
    predictor,
    sup_edge_types,
    epoch,
    val_loss,
    config,
    *,
    filename="best_model.pt",
    data_metadata=None,
    train_pos=None,
    val_pos=None,
    test_pos=None,
    loader_params=None,
    torch_seed=None,
    optimizer_state=None,
    patience_counter=None,
    best_val=None,
    best_model_state=None,
    best_predictor_state=None,
    training_params=None,
):
    """
    Save a checkpoint containing model/predictor weights and run metadata.
    Optionally include splits and loader params so evaluation notebooks
    can rebuild loaders without rerunning the split step.
    """
    save_path = get_models_dir() / filename
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "sup_edge_types": sup_edge_types,
            "config": config,
            "data_metadata": data_metadata,
            "train_pos": train_pos,
            "val_pos": val_pos,
            "test_pos": test_pos,
            "loader_params": loader_params,
            "torch_seed": torch_seed,
            "optimizer_state": optimizer_state,
            "patience_counter": patience_counter,
            "best_val": best_val,
            "best_model_state_dict": best_model_state,
            "best_predictor_state_dict": best_predictor_state,
            "training_params": training_params,
        },
        save_path,
    )
    return save_path


def _build_model_from_checkpoint(checkpoint, device, metadata_override=None):
    config = checkpoint.get("config", {})
    hidden = config.get("hidden", 128)
    out_dim = config.get("out_dim", 128)
    layers = config.get("layers", 2)
    dropout = config.get("dropout", 0.1)
    score_head = config.get("score_head", "dot")

    metadata = metadata_override or checkpoint.get("data_metadata")
    if metadata is None:
        raise ValueError("No metadata provided or stored in checkpoint to rebuild the model.")

    model = HeteroSAGE(
        metadata=metadata,
        hidden=hidden,
        out=out_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    predictor = MLPredictor(out_dim).to(device) if score_head == "mlp" else DotPredictor().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    return model, predictor


def load_model_checkpoint(device=None, metadata=None, filename="best_model.pt"):
    """
    Load a checkpoint from disk and rebuild the model/predictor with stored config.
    Device and metadata are optional; the function will auto-pick a device and use
    checkpoint metadata when available.
    """
    device = select_device(device)
    load_path = get_models_dir() / filename
    checkpoint = torch.load(load_path, map_location=device)
    model, predictor = _build_model_from_checkpoint(checkpoint, device, metadata_override=metadata)
    return model, predictor, checkpoint


def load_full_run(data, device=None, filename="best_model.pt"):
    """
    Load model/predictor plus splits and rebuild loaders from a saved checkpoint.
    Requires the full graph data to rebuild loaders; falls back to CPU if no GPU/MPS.
    """
    data_cpu = data.to('cpu')
    device = select_device(device)
    # Use checkpoint-stored metadata to avoid mismatches with the current data object.
    model, predictor, checkpoint = load_model_checkpoint(device=device, metadata=None, filename=filename)

    train_pos = checkpoint.get("train_pos")
    val_pos = checkpoint.get("val_pos")
    test_pos = checkpoint.get("test_pos")
    loader_params = checkpoint.get("loader_params") or {}
    sup_edge_types = checkpoint.get("sup_edge_types")

    if train_pos is None or val_pos is None or test_pos is None or sup_edge_types is None:
        raise ValueError("Checkpoint does not contain saved splits/edge types; cannot rebuild loaders.")

    neg_ratio = loader_params.get("neg_ratio", 1.0)
    batch_size = loader_params.get("batch_size", 1024)
    fanout = loader_params.get("fanout", [15, 10])

    train_graph = data_cpu.clone()
    for et in sup_edge_types:
        train_graph[et].edge_index = train_pos[et]

    loaders = make_link_loaders(
        train_graph=train_graph,
        full_graph=data_cpu,
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        edge_types=sup_edge_types,
        neg_ratio=neg_ratio,
        batch_size=batch_size,
        fanout=fanout,
    )

    splits = {
        "train_graph": train_graph,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "sup_ets": sup_edge_types,
    }

    return model, predictor, loaders, splits, checkpoint


def load_training_state(data, device=None, filename="best_model.pt"):
    """
    Load everything needed to resume training: model, predictor, optimizer state,
    splits, loaders, and training metadata.
    """
    data_cpu = data.to('cpu')
    device = select_device(device)
    checkpoint = torch.load(get_models_dir() / filename, map_location=device)

    model, predictor = _build_model_from_checkpoint(checkpoint, device, metadata_override=None)

    train_pos = checkpoint.get("train_pos")
    val_pos = checkpoint.get("val_pos")
    test_pos = checkpoint.get("test_pos")
    sup_edge_types = checkpoint.get("sup_edge_types")
    loader_params = checkpoint.get("loader_params") or {}

    if train_pos is None or val_pos is None or test_pos is None or sup_edge_types is None:
        raise ValueError("Checkpoint does not contain saved splits/edge types; cannot rebuild loaders.")

    neg_ratio = loader_params.get("neg_ratio", 1.0)
    batch_size = loader_params.get("batch_size", 1024)
    fanout = loader_params.get("fanout", [15, 10])

    train_graph = data_cpu.clone()
    for et in sup_edge_types:
        train_graph[et].edge_index = train_pos[et]

    loaders = make_link_loaders(
        train_graph=train_graph,
        full_graph=data_cpu,
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
        edge_types=sup_edge_types,
        neg_ratio=neg_ratio,
        batch_size=batch_size,
        fanout=fanout,
    )

    training_params = checkpoint.get("training_params") or {}
    lr = training_params.get("lr", 1e-3)
    wd = training_params.get("wd", 1e-4)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    opt_state = checkpoint.get("optimizer_state")
    if opt_state:
        opt.load_state_dict(opt_state)

    best_val = checkpoint.get("best_val", checkpoint.get("val_loss", float("inf")))
    patience_counter = checkpoint.get("patience_counter", 0)
    start_epoch = checkpoint.get("epoch", 0)
    best_state = {
        "model": checkpoint.get("best_model_state_dict", checkpoint.get("model_state_dict")),
        "pred": checkpoint.get("best_predictor_state_dict", checkpoint.get("predictor_state_dict")),
    }

    splits = {
        "train_graph": train_graph,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "sup_ets": sup_edge_types,
    }

    return (
        model,
        predictor,
        opt,
        loaders,
        splits,
        checkpoint,
        start_epoch,
        patience_counter,
        best_val,
        best_state,
    )
