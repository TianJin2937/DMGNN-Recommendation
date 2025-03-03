# DMGNN: A Dual Multi-Relational GNN Model for Enhanced Recommendation

**Paper**: [Computers, Materials & Continua, Vol. 84(2), 2025](https://www.sciencedirect.com/org/science/article/pii/S1546221825006058) | 1 Citation

## Abstract

This paper proposes a dual multi-relational graph neural network recommendation algorithm (DMGNN) that constructs two complementary graph structures: a User-Item Interaction Graph (UIIG) and a Relational Association Graph (RAG). The model features two parallel GCN branches with an adaptive attention-based fusion mechanism, achieving up to 12.3% improvement in Precision and 15.2% boost in recommendation diversity.

## Datasets

- [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)
- [Amazon-Electronics](https://jmcauley.ucsd.edu/data/amazon/)
- [Yelp](https://www.yelp.com/dataset)

## Methods

- **User-Item Interaction Graph (UIIG)** — Direct user-item interactions weighted by frequency
- **Relational Association Graph (RAG)** — Latent associations via user/item similarity (cosine)
- **Dual GCN Branches** — Parallel multi-layer graph convolutions on both graphs
- **Attention Fusion** — Learned weights to adaptively combine branch embeddings
- **Recommendation** — Cosine similarity ranking on final embeddings

## Repository Structure

```
├── src/
│   ├── model.py              # DMGNN model architecture
│   ├── graph_construction.py # UIIG and RAG graph builders
│   ├── train.py              # Training loop
│   └── evaluate.py           # Precision@N, Recall@N, F1@N, ILS
├── configs/
│   └── default.yaml          # Hyperparameters
├── requirements.txt
└── README.md
```

## Results

| Algorithm | Precision@10 | Recall@10 | F1@10 |
|-----------|-------------|-----------|-------|
| User-Based CF | 0.08 | 0.25 | 0.12 |
| Item-Based CF | 0.10 | 0.28 | 0.14 |
| MF | 0.13 | 0.30 | 0.16 |
| NGCF | 0.15 | 0.33 | 0.20 |
| LightGCN | 0.14 | 0.31 | 0.18 |
| **DMGNN** | **0.35** | **0.40** | **0.38** |

## Citation

```bibtex
@article{li2025dmgnn,
  title={DMGNN: A Dual Multi-Relational GNN Model for Enhanced Recommendation},
  author={Li, Siyue and Jin, Tian and Wang, Erfan and Tao, Ranting and Lu, Jiaxin and Xi, Kai},
  journal={Computers, Materials and Continua},
  volume={84},
  number={2},
  pages={2331--2353},
  year={2025},
  doi={10.32604/cmc.2025.066382}
}
```

## Requirements

```
torch>=1.12
torch-geometric>=2.2
numpy
pandas
scipy
scikit-learn
pyyaml
```

## License

MIT
