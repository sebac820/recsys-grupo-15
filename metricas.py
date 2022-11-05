from funciones_auxiliares import (
    average_precision, dcg_usuario, probabilidad_de_conocer_dado_item,
    probabilidad_de_item_dada_lista_de_recomendaciones
)
from math import log2


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
        recomendaciones_usuario = recomendaciones[usuario]
        suma_ndcgs += dcg_usuario(top_n_verdadero, recomendaciones_usuario, arroba) / idcg
    # Obtener NDCG promedio:
    return suma_ndcgs / len(top_n_verdadero_por_usuario)


# MAP


def mean_average_precision(top_n_verdadero_por_usuario: dict, recomendaciones: dict, arroba: int):
    suma_de_average_precisions = 0
    for usuario, top_n_verdadero in top_n_verdadero_por_usuario.items():
        suma_de_average_precisions += average_precision(top_n_verdadero, recomendaciones[usuario], arroba)
    return suma_de_average_precisions / len(top_n_verdadero_por_usuario)


# Novedad


def novedad(recomendaciones, items, interacciones):
    novedad = 0
    for item in recomendaciones:
        novedad += (
            probabilidad_de_item_dada_lista_de_recomendaciones(item, recomendaciones, items, interacciones)
            *
            (1 - probabilidad_de_conocer_dado_item(item))
        )
    return novedad


# Diversidad


def diversidad(recomendaciones):
    distancias = 0
    for n in range(1, len(recomendaciones)):
        for k in range(0, n):
            distancias += distancia(recomendaciones[n], recomendaciones[k])
    return 2 * distancias / (len(recomendaciones) * (len(recomendaciones) - 1))
