"""Evaluation metrics: Precision@N, Recall@N, F1@N, Intra-list Similarity."""
import numpy as np
import torch
import torch.nn.functional as F


def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def f1_at_k(recommended, relevant, k):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def intra_list_similarity(item_embeds, recommended_ids):
    """Measure diversity: lower ILS = more diverse recommendations."""
    if len(recommended_ids) < 2:
        return 0.0
    embeds = F.normalize(item_embeds[recommended_ids], dim=-1)
    sim_matrix = torch.mm(embeds, embeds.t())
    n = len(recommended_ids)
    # Average pairwise similarity (excluding diagonal)
    ils = (sim_matrix.sum() - n) / (n * (n - 1))
    return ils.item()


def evaluate(model, uiig_edge_index, rag_edge_index, uiig_edge_weight,
             rag_edge_weight, test_data, k_values=[5, 10, 20]):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        user_embeds, item_embeds = model(uiig_edge_index, rag_edge_index,
                                         uiig_edge_weight, rag_edge_weight)

    results = {k: {'precision': [], 'recall': [], 'f1': []} for k in k_values}

    for user_id, relevant_items in test_data.items():
        scores = model.predict(user_embeds, item_embeds,
                              torch.full((item_embeds.size(0),), user_id, dtype=torch.long),
                              torch.arange(item_embeds.size(0)))
        _, top_indices = torch.topk(scores, max(k_values))
        recommended = top_indices.cpu().numpy().tolist()

        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant_items, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant_items, k))
            results[k]['f1'].append(f1_at_k(recommended, relevant_items, k))

    for k in k_values:
        print(f"Precision@{k}: {np.mean(results[k]['precision']):.4f}")
        print(f"Recall@{k}: {np.mean(results[k]['recall']):.4f}")
        print(f"F1@{k}: {np.mean(results[k]['f1']):.4f}")

    return results
