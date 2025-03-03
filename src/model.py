"""DMGNN: Dual Multi-Relational Graph Neural Network for Recommendation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DMGNNBranch(nn.Module):
    """Single GCN branch for either UIIG or RAG."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class DMGNN(nn.Module):
    """Dual Multi-Relational Graph Neural Network.

    Two parallel branches (UIIG + RAG) with attention-based fusion.
    """

    def __init__(self, num_users, num_items, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # Embeddings
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # Dual branches
        self.uiig_branch = DMGNNBranch(embed_dim, hidden_dim, embed_dim, num_layers)
        self.rag_branch = DMGNNBranch(embed_dim, hidden_dim, embed_dim, num_layers)

        # Attention fusion
        self.attn_uiig = nn.Linear(embed_dim, 1)
        self.attn_rag = nn.Linear(embed_dim, 1)

        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, uiig_edge_index, rag_edge_index, uiig_edge_weight=None, rag_edge_weight=None):
        # Initial embeddings
        x = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)

        # Dual branch forward
        h_uiig = self.uiig_branch(x, uiig_edge_index, uiig_edge_weight)
        h_rag = self.rag_branch(x, rag_edge_index, rag_edge_weight)

        # Attention-based fusion
        alpha_uiig = torch.sigmoid(self.attn_uiig(h_uiig))
        alpha_rag = torch.sigmoid(self.attn_rag(h_rag))
        alpha_sum = alpha_uiig + alpha_rag
        alpha_uiig = alpha_uiig / alpha_sum
        alpha_rag = alpha_rag / alpha_sum

        h_fused = alpha_uiig * h_uiig + alpha_rag * h_rag

        user_embeds = h_fused[:self.num_users]
        item_embeds = h_fused[self.num_users:]
        return user_embeds, item_embeds

    def predict(self, user_embeds, item_embeds, user_ids, item_ids):
        """Cosine similarity for recommendation scoring."""
        u = F.normalize(user_embeds[user_ids], dim=-1)
        i = F.normalize(item_embeds[item_ids], dim=-1)
        return (u * i).sum(dim=-1)
