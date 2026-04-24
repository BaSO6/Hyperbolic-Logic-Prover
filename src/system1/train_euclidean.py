# Modification 1: Import Euclidean model
from src.system1.models_euclidean import EuclideanGCN

# Modification 2: Initialize model
model = EuclideanGCN(num_features=384, hidden_dim=128, output_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Euclidean space typically does not require RiemannianAdam

# Modification 3: Loss Function (Using Euclidean distance/cosine similarity)
def euclidean_contrastive_loss(embeddings, positives, negatives, margin=1.0):
    """
    Positives: Distance should be small
    Negatives: Distance should be large
    """
    # L2 Distance
    pos_dist = torch.norm(embeddings[positives[:, 0]] - embeddings[positives[:, 1]], p=2, dim=-1)
    neg_dist = torch.norm(embeddings[negatives[:, 0]] - embeddings[negatives[:, 1]], p=2, dim=-1)
    
    # Margin Loss: max(0, pos - neg + margin)
    loss = torch.relu(pos_dist - neg_dist + margin).mean()
    return loss

# Alternatively, use InfoNCE (More modern, recommended)
def infonce_loss(anchor, positive, negatives, temperature=0.1):
    # Cosine Similarity
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    neg_sim = torch.matmul(anchor, negatives.T) / temperature
    
    # LogSoftmax
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# Call within the Training Loop
loss = infonce_loss(premise_emb, conclusion_emb, batch_negatives)