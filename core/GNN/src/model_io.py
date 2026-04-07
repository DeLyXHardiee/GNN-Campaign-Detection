import torch
from pathlib import Path

from .model import HeteroSAGE, DotPredictor, MLPredictor, DistMultPredictor
from .loaders import make_link_loaders

_CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt"}


def get_models_dir() -> Path:
    """
    Return the directory where model checkpoints are stored.
    Creates the directory if it does not exist.
    """
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def _validated_checkpoint_path(filename) -> Path:
    candidate = Path(filename).expanduser()
    path = candidate if candidate.is_file() else (get_models_dir() / str(filename))
    path = path.resolve()
    if path.suffix.lower() not in _CHECKPOINT_EXTS:
        raise ValueError(f"Unsupported checkpoint extension for {path}.")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def select_device(preferred=None):
    """
    Auto-pick a device (cuda > cpu) unless a preferred one is provided.
    """
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model_checkpoint(
    model,
    predictor,
    sup_edge_types,
    epoch,
    val_loss,
    config,
    *,
    save_dir=None,
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
    projector=None,
    training_objective="link_prediction",
):
    """
    Save a checkpoint containing model/predictor weights and run metadata.
    Optionally include splits and loader params so evaluation notebooks
    can rebuild loaders without rerunning the split step.
    """
    base_dir = Path(save_dir) if save_dir is not None else get_models_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / filename
    payload = {
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
        "training_objective": training_objective,
    }
    if projector is not None:
        payload["projector_state_dict"] = projector.state_dict()
    torch.save(payload, save_path)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    return save_path


def save_vicreg_checkpoint(
    *,
    save_dir,
    filename: str,
    model,
    projector,
    optimizer,
    epoch: int,
    patience_counter: int,
    encoder_config: dict,
    data_metadata,
    torch_seed: int,
    email_train_idx,
    email_val_idx,
    email_test_idx,
    vicreg_hparams: dict,
    anchor_loader_params: dict,
    optimizer_state_dict,
    val_vicreg_total: float,
    best_val_vicreg_total: float,
) -> Path:
    """
    Save a VICReg-only checkpoint (no link-prediction tensors or predictor weights).
    ``model_state_dict`` / ``projector_state_dict`` are the weights at this save (best file = best-so-far).
    """
    base_dir = Path(save_dir) if save_dir is not None else get_models_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / filename
    payload = {
        "training_objective": "vicreg",
        "epoch": epoch,
        "val_vicreg_total": float(val_vicreg_total),
        "best_val_vicreg_total": float(best_val_vicreg_total),
        "patience_counter": int(patience_counter),
        "model_state_dict": model.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "encoder_config": encoder_config,
        "data_metadata": data_metadata,
        "torch_seed": torch_seed,
        "email_train_idx": email_train_idx,
        "email_val_idx": email_val_idx,
        "email_test_idx": email_test_idx,
        "vicreg_hparams": vicreg_hparams,
        "anchor_loader_params": anchor_loader_params,
    }
    torch.save(payload, save_path)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    return save_path


def save_contrastive_checkpoint(
    *,
    save_dir,
    filename: str,
    model,
    projector,
    optimizer,
    epoch: int,
    patience_counter: int,
    encoder_config: dict,
    data_metadata,
    torch_seed: int,
    email_train_idx,
    email_val_idx,
    email_test_idx,
    contrastive_hparams: dict,
    anchor_loader_params: dict,
    optimizer_state_dict,
    val_contrastive_total: float,
    best_val_contrastive_total: float,
) -> Path:
    """Save a contrastive-only checkpoint (encoder + projector, email anchor splits)."""
    base_dir = Path(save_dir) if save_dir is not None else get_models_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / filename
    payload = {
        "training_objective": "contrastive",
        "epoch": epoch,
        "val_contrastive_total": float(val_contrastive_total),
        "best_val_contrastive_total": float(best_val_contrastive_total),
        "patience_counter": int(patience_counter),
        "model_state_dict": model.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "encoder_config": encoder_config,
        "data_metadata": data_metadata,
        "torch_seed": torch_seed,
        "email_train_idx": email_train_idx,
        "email_val_idx": email_val_idx,
        "email_test_idx": email_test_idx,
        "contrastive_hparams": contrastive_hparams,
        "anchor_loader_params": anchor_loader_params,
    }
    torch.save(payload, save_path)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    return save_path


def _build_vicreg_encoder_from_checkpoint(checkpoint, device, metadata_override=None):
    """Rebuild HeteroSAGE from a VICReg checkpoint. Returns (model, None) for projector slot."""
    enc_cfg = checkpoint.get("encoder_config") or checkpoint.get("config") or {}
    hidden = enc_cfg.get("hidden", 128)
    out_dim = enc_cfg.get("out_dim", 128)
    layers = enc_cfg.get("layers", 2)
    dropout = enc_cfg.get("dropout", 0.3)
    metadata = metadata_override or checkpoint.get("data_metadata")
    if metadata is None:
        raise ValueError("VICReg checkpoint missing data_metadata; cannot rebuild encoder.")
    model = HeteroSAGE(
        metadata=metadata,
        hidden=hidden,
        out=out_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictor = DotPredictor().to(device)
    return model, predictor


def load_vicreg_encoder_checkpoint(device=None, metadata=None, filename="best_model.pt"):
    """
    Load only the hetero encoder from a VICReg checkpoint (for embedding extraction / clustering).

    Returns ``(model, checkpoint)`` where ``model`` is ``HeteroSAGE``. The checkpoint dict
    includes ``projector_state_dict`` if you need the training projector.
    """
    device = select_device(device)
    load_path = _validated_checkpoint_path(filename)
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        load_path, map_location=device, weights_only=True
    )
    if checkpoint.get("training_objective") != "vicreg":
        raise ValueError(
            f"Expected training_objective='vicreg' in checkpoint, got {checkpoint.get('training_objective')!r}."
        )
    model, _pred = _build_vicreg_encoder_from_checkpoint(
        checkpoint, device, metadata_override=metadata
    )
    return model, checkpoint


def load_contrastive_encoder_checkpoint(device=None, metadata=None, filename="best_model.pt"):
    """
    Load the hetero encoder from a contrastive training checkpoint (embedding extraction / clustering).
    """
    device = select_device(device)
    load_path = _validated_checkpoint_path(filename)
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        load_path, map_location=device, weights_only=True
    )
    if checkpoint.get("training_objective") != "contrastive":
        raise ValueError(
            f"Expected training_objective='contrastive' in checkpoint, got {checkpoint.get('training_objective')!r}."
        )
    model, _pred = _build_vicreg_encoder_from_checkpoint(
        checkpoint, device, metadata_override=metadata
    )
    return model, checkpoint


def _build_model_from_checkpoint(checkpoint, device, metadata_override=None):
    obj = checkpoint.get("training_objective")
    if obj in ("vicreg", "contrastive"):
        return _build_vicreg_encoder_from_checkpoint(checkpoint, device, metadata_override)

    config = checkpoint.get("config", {})
    hidden = config.get("hidden", 256)
    out_dim = config.get("out_dim", 256)
    layers = config.get("layers", 2)
    dropout = config.get("dropout", 0.3)
    score_head = config.get("score_head", "dot")
    predictor_hidden = config.get("predictor_hidden")

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
    if score_head == "mlp":
        pred_state = checkpoint["predictor_state_dict"]
        hidden_size = predictor_hidden or pred_state["net.0.weight"].shape[0]
        predictor = MLPredictor(out_dim, h=hidden_size).to(device)
    elif score_head == "distmult":
        sup_edge_types = checkpoint.get("sup_edge_types")
        if sup_edge_types is None:
            raise ValueError("Checkpoint missing sup_edge_types needed for DistMultPredictor.")
        predictor = DistMultPredictor(dim=out_dim, edge_types=sup_edge_types).to(device)
    else:
        predictor = DotPredictor().to(device)
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
    load_path = _validated_checkpoint_path(filename)
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        load_path, map_location=device, weights_only=True
    )
    model, predictor = _build_model_from_checkpoint(checkpoint, device, metadata_override=metadata)
    return model, predictor, checkpoint


def load_full_run(data, device=None, filename="best_model.pt"):
    """
    Load model/predictor plus splits and rebuild loaders from a saved checkpoint.
    Requires the full graph data to rebuild loaders; falls back to CPU if no GPU/MPS.
    """
    data_cpu = data.to('cpu')
    device = select_device(device)
    load_path = _validated_checkpoint_path(filename)
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        load_path, map_location=device, weights_only=True
    )
    if checkpoint.get("training_objective") in ("vicreg", "contrastive"):
        raise ValueError(
            "Checkpoint is self-supervised (VICReg/contrastive); load_full_run (link loaders) does not apply. "
            "Use load_vicreg_encoder_checkpoint() / load_contrastive_encoder_checkpoint(), "
            "or NeighborLoader + email splits from the checkpoint."
        )
    model, predictor = _build_model_from_checkpoint(checkpoint, device, metadata_override=None)

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
    load_path = _validated_checkpoint_path(filename)
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        load_path, map_location=device, weights_only=True
    )
    if checkpoint.get("training_objective") in ("vicreg", "contrastive"):
        raise ValueError(
            "Checkpoint is self-supervised (VICReg/contrastive); load_training_state (LP loaders + resume) is not supported. "
            "Use the appropriate encoder loader for inference, or add a dedicated resume path."
        )

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
