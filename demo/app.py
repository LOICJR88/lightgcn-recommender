import torch
import pandas as pd
import gradio as gr

data = torch.load("embeddings.pt", map_location="cpu")
item_emb = torch.nn.functional.normalize(data["item_emb"], dim=1)  # normalisé pour cosinus
item_idx_to_movieId = data["item_idx_to_movieId"]
movieId_to_item_idx = {v: k for k, v in item_idx_to_movieId.items()}

movies = pd.read_csv("movies.csv")
movieId_to_title = dict(zip(movies["movieId"], movies["title"]))

# Liste des titres pour le menu déroulant (uniquement les films présents dans le modèle)
available_titles = sorted([
    movieId_to_title[mid] for mid in item_idx_to_movieId.values()
    if mid in movieId_to_title
])
title_to_movieId = {v: k for k, v in movieId_to_title.items()}


def recommend(movie1, movie2, movie3, top_k):
    chosen_titles = [m for m in [movie1, movie2, movie3] if m]
    if not chosen_titles:
        return "Sélectionne au moins un film."

    chosen_indices = []
    for title in chosen_titles:
        mid = title_to_movieId[title]
        chosen_indices.append(movieId_to_item_idx[mid])

    # Profil = moyenne des embeddings des films choisis
    profile = item_emb[chosen_indices].mean(dim=0)
    profile = torch.nn.functional.normalize(profile, dim=0)

    scores = item_emb @ profile
    for idx in chosen_indices:
        scores[idx] = -float("inf")  # ne pas recommander les films déjà choisis

    top_items = torch.topk(scores, int(top_k)).indices.tolist()

    results = []
    for rank, item_idx in enumerate(top_items, 1):
        mid = item_idx_to_movieId[item_idx]
        title = movieId_to_title.get(mid, f"Film #{mid}")
        results.append(f"{rank}. {title}")

    return "\n".join(results)


demo = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Dropdown(choices=available_titles, label="Film aimé n°1", allow_custom_value=False),
        gr.Dropdown(choices=available_titles, label="Film aimé n°2 (optionnel)"),
        gr.Dropdown(choices=available_titles, label="Film aimé n°3 (optionnel)"),
        gr.Slider(5, 20, value=10, step=1, label="Nombre de recommandations")
    ],
    outputs=gr.Textbox(label="Films recommandés"),
    title="LightGCN — Recommandation de films (MovieLens)",
    description="Sélectionne 1 à 3 films que tu aimes, et reçois des recommandations basées sur leurs embeddings appris par LightGCN."
)

demo.launch()
