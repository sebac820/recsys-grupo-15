from funciones_auxiliares import (
    average_precision, dcg_usuario, probabilidad_de_que_item_sea_conocido_por_usuario
)
from math import log2
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


def novelty_for_single_user(recomendaciones, usuario, interacciones):
    novedades = 0
    for item in recomendaciones:
        novedades += 1 - probabilidad_de_que_item_sea_conocido_por_usuario(item, usuario, interacciones)
    return novedades / len(recomendaciones)


def novelty_for_multiple_users(recommendations, interactions):
    novelty = 0
    for user, recommended_items in recommendations:
        novelty += novelty_for_single_user(recommended_items, user, interactions)
    return novelty / len(recommendations)


# Diversidad


def diversity_for_single_user(recommendations: list, item_features):
    distancias = 0
    for n in range(1, len(recommendations)):
        item_n = item_features[item_features['track_id'] == recommendations[n]].drop(columns=['track_id'])
        for k in range(0, n):
            item_k = item_features[item_features['track_id'] == recommendations[k]].drop(columns=['track_id'])
            distancias += scipy.spatial.distance.cosine(item_n, item_k)
    return 2 * distancias / (len(recommendations) * (len(recommendations) - 1))


def diversity_for_multiple_users(recommendations: dict, item_features):
    diversity = 0
    for recommended_items in recommendations.values():
        diversity += diversity_for_single_user(recommended_items, item_features)
    return diversity / len(recommendations)
