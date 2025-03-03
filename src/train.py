"""Training loop for DMGNN."""
import torch
import torch.nn.functional as F
from torch.optim import Adam


def train_epoch(model, optimizer, uiig_edge_index, rag_edge_index,
                uiig_edge_weight, rag_edge_weight, train_pairs, num_items):
    """BPR loss training for one epoch."""
    model.train()
    optimizer.zero_grad()

    user_embeds, item_embeds = model(uiig_edge_index, rag_edge_index,
                                     uiig_edge_weight, rag_edge_weight)

    users, pos_items, neg_items = train_pairs
    pos_scores = model.predict(user_embeds, item_embeds, users, pos_items)
    neg_scores = model.predict(user_embeds, item_embeds, users, neg_items)

    # BPR loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    # L2 regularization
    reg_loss = 0.01 * (user_embeds[users].norm(2).pow(2) +
                       item_embeds[pos_items].norm(2).pow(2) +
                       item_embeds[neg_items].norm(2).pow(2)) / len(users)

    total_loss = loss + reg_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train(model, uiig_edge_index, rag_edge_index, uiig_edge_weight,
          rag_edge_weight, train_pairs, num_items, epochs=200, lr=1e-3):
    """Full training loop."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        loss = train_epoch(model, optimizer, uiig_edge_index, rag_edge_index,
                          uiig_edge_weight, rag_edge_weight, train_pairs, num_items)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    return model
