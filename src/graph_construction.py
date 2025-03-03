"""Graph construction for UIIG and RAG."""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_uiig(interactions, num_users, num_items):
    """Build User-Item Interaction Graph with weighted edges.

    Edge weight: w_ui = count(u,i) / (sqrt(|N_u|) * sqrt(|N_i|))
    """
    user_ids, item_ids, counts = [], [], []
    user_counts = np.zeros(num_users)
    item_counts = np.zeros(num_items)

    for u, i, c in interactions:
        user_counts[u] += c
        item_counts[i] += c

    edges_src, edges_dst, weights = [], [], []
    for u, i, c in interactions:
        w = c / (np.sqrt(user_counts[u]) * np.sqrt(item_counts[i]) + 1e-8)
        # Bidirectional edges (user->item and item->user)
        edges_src.extend([u, i + num_users])
        edges_dst.extend([i + num_users, u])
        weights.extend([w, w])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


def build_rag(interactions, num_users, num_items, sim_threshold=0.2):
    """Build Relational Association Graph based on user/item similarity.

    User similarity: cosine similarity of interaction vectors.
    Item similarity: cosine similarity of user-interaction vectors.
    """
    # Build interaction matrix
    rows, cols, vals = [], [], []
    for u, i, c in interactions:
        rows.append(u)
        cols.append(i)
        vals.append(c)
    mat = csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))

    # User similarity
    user_sim = cosine_similarity(mat)
    # Item similarity
    item_sim = cosine_similarity(mat.T)

    edges_src, edges_dst, weights = [], [], []

    # User-user edges
    for u in range(num_users):
        sims = user_sim[u]
        neighbors = np.where(sims > sim_threshold)[0]
        for v in neighbors:
            if v != u:
                edges_src.append(u)
                edges_dst.append(v)
                weights.append(sims[v])

    # Item-item edges (offset by num_users)
    for i in range(num_items):
        sims = item_sim[i]
        neighbors = np.where(sims > sim_threshold)[0]
        for j in neighbors:
            if j != i:
                edges_src.append(i + num_users)
                edges_dst.append(j + num_users)
                weights.append(sims[j])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight
