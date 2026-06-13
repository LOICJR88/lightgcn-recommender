# LightGCN — Système de Recommandation sur MovieLens

Implémentation du modèle **LightGCN** (Light Graph Convolutional Network)
pour un système de recommandation de films basé sur le filtrage collaboratif.

## Description

Ce projet implémente LightGCN, une architecture GCN simplifiée qui supprime
les transformations non-linéaires pour le filtrage collaboratif à grande échelle.
Le dataset utilisé est MovieLens ml-latest-small (~100 000 évaluations).

## Architecture

- **Modèle** : LightGCN avec 3 couches de propagation, embeddings de dimension 64
- **Entraînement** : Perte BPR, optimiseur Adam, 300 époques
- **Évaluation** : Recall@K et NDCG@K (K = 10, 20)

## Résultats

| Modèle      | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|-------------|-----------|---------|-----------|---------|
| LightGCN    | 0.0268    | 0.0229  | 0.0538    | 0.0349  |
| Popularité  | 0.0189    | 0.0126  | 0.0427    | 0.0221  |

LightGCN améliore la baseline de **+42% sur le NDCG@10**.

Voir `results/metrics.csv` pour le tableau complet et `figures/` pour la
courbe d'apprentissage et la visualisation t-SNE des embeddings.

## Démo interactive

Une démo Gradio permet de sélectionner 1 à 3 films aimés et d'obtenir des
recommandations basées sur la similarité des embeddings appris par LightGCN.

🔗 **Démo en ligne** : [À COMPLÉTER - lien Hugging Face Spaces]

Code source : dossier `demo/`

## Structure du projet
lightgcn-recommender/

├── LightGCN_MovieLens.ipynb     # Notebook principal (code source)

├── best_model.pt                # Modèle entraîné (poids)

├── embeddings.pt                # Embeddings exportés (pour la démo)

├── figures/

│   ├── loss_curve.png

│   └── tsne_embeddings.png

├── results/

│   └── metrics.csv

├── demo/

│   ├── app.py

│   ├── movies.csv

│   └── requirements.txt

├── requirements.txt

└── README.md
## Installation et utilisation

### Option 1 — Google Colab (recommandé)
Ouvrir directement le notebook dans Google Colab.

### Option 2 — Local
```bash
pip install -r requirements.txt
jupyter notebook LightGCN_MovieLens.ipynb
```

### Lancer la démo localement
```bash
cd demo
pip install -r requirements.txt
python app.py
```

## Points théoriques clés

- **Graphe biparti** : représentation des interactions utilisateur-film
- **LightGCN vs GCN classique** : suppression des matrices de transformation W et des biais
- **Perte BPR** : optimisation par paires positives/négatives
- **Agrégation multicouche** : moyenne des embeddings de chaque couche
- **Métriques de ranking** : Recall et NDCG

## Dataset

MovieLens ml-latest-small :
- 100 836 évaluations
- 610 utilisateurs
- 9 724 films

Téléchargement automatique dans le notebook.

## Environnement

- Python 3.10+
- PyTorch 2.x
- GPU T4 (Google Colab)
