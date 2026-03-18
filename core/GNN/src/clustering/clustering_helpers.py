@torch.no_grad()
def extract_email_embeddings(model, data, device):
    model.eval()
    x_dict = data.to(device).x_dict
    edge_index_dict = data.to(device).edge_index_dict
    h = model(x_dict, edge_index_dict)
    email_vecs = h['email'].cpu().numpy()
    email_ids = np.arange(len(email_vecs))  # Assumes IDs are ordered
    return email_vecs, email_ids

