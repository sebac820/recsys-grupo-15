from funciones_auxiliares import (
    average_precision, dcg_usuario, probabilidad_de_que_item_sea_conocido_por_usuario
)
from math import log2
import numpy as np
import pandas as pd
import scipy.spatial.distance


# NDCG


def ndcg(top_n_verdadero_por_usuario: dict, recomendaciones: dict, arroba: int):
    # Inicializar suma de NDCGs:
    suma_ndcgs = 0
    # Calcular IDCG:
    idcg = 0
    for i in range(1, arroba+1):
        idcg += 1 / log2(i + 1)
    # Para cada usuario, calcular su NDCG y agregarlo a la suma de NDCGs:
    for usuario, top_n_verdadero in top_n_verdadero_por_usuario.items():
        suma_ndcgs += dcg_usuario(top_n_verdadero, recomendaciones[usuario], arroba) / idcg
    # Retornar NDCG promedio:
    return suma_ndcgs / len(top_n_verdadero_por_usuario)


# MAP


def mean_average_precision(top_n_verdadero_por_usuario: dict, recomendaciones: dict, arroba: int):
    suma_de_average_precisions = 0
    for usuario, top_n_verdadero in top_n_verdadero_por_usuario.items():
        suma_de_average_precisions += average_precision(top_n_verdadero, recomendaciones[usuario], arroba)
    return suma_de_average_precisions / len(top_n_verdadero_por_usuario)


# Novedad


def novelty_for_single_user(user_id: str, recommended_items: np.ndarray, interactions: pd.DataFrame, item_features: pd.DataFrame):
    user_history = np.array(interactions[interactions['user_id'] == user_id]['track_id'])
    recommended_items = np.array(recommended_items)
    novelty = 0
    for recommended_item_id in recommended_items:
        recommended_item_features = item_features.loc[recommended_item_id]
        for seen_item_id in user_history:
            seen_item_features = item_features.loc[seen_item_id]
            novelty += scipy.spatial.distance.cosine(recommended_item_features, seen_item_features)
    return novelty / (len(recommended_items) * len(user_history))


def novelty_for_multiple_users(users_and_recommendations: dict, interactions: pd.DataFrame, item_features: pd.DataFrame):
    novelty = 0
    for user_id, recommended_items in users_and_recommendations.items():
        novelty += novelty_for_single_user(user_id, recommended_items, interactions, item_features)
    return novelty / len(users_and_recommendations)


# Diversidad


def diversity_for_single_user(recommendations: list, item_features: pd.DataFrame):
    len_recommendations = len(recommendations)
    distances = 0
    for n in np.arange(1, len_recommendations):
        item_n = item_features.loc[recommendations[n]]
        for k in np.arange(0, n):
            distances += scipy.spatial.distance.cosine(item_n, item_features[recommendations[k]])
    return 2 * distances / (len_recommendations * (len_recommendations - 1))


def diversity_for_multiple_users(recommendations: dict, item_features):
    diversity = 0
    for recommended_items in recommendations.values():
        diversity += diversity_for_single_user(recommended_items, item_features)
    return diversity / len(recommendations)
