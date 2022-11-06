from base64 import b64encode
from math import log2
import matplotlib.pyplot as plt
import scipy
import json
import requests


def average_precision(top_n_verdadero: list, recomendaciones: list, arroba: int):
    # Evitar "index out of range":
    limite_top_n_verdadero = min(arroba, len(top_n_verdadero))
    # Inicializar variables en cero:
    verdaderos_positivos_hasta_el_momento = 0
    suma_de_precisiones = 0
    # Para cada ítem dentro del límite dado por @:
    for i in range(1, arroba+1):
        item = recomendaciones[i-1]
        # Si el ítem está en el top_n_verdadero, entonces es relevante
        if item in top_n_verdadero[0:limite_top_n_verdadero]:
            verdaderos_positivos_hasta_el_momento += 1
            suma_de_precisiones += verdaderos_positivos_hasta_el_momento / i
    # Retornar precisión promedio:
    if verdaderos_positivos_hasta_el_momento == 0:
        return 0
    return suma_de_precisiones / verdaderos_positivos_hasta_el_momento
    # Fórmula complementada por
    # https://medium.com/@misty.mok/how-mean-average-precision-at-k-map-k-can-be-more-useful-than-
    # other-evaluation-metrics-6881e0ee21a9


def dcg_usuario(top_n_verdadero: list, recomendaciones: list, arroba: int):
    # Evitar errores de "index out of range":
    limite_top_n_verdadero = min(arroba, len(top_n_verdadero))
    # Inicializar la suma de gains en cero:
    dcg = 0
    # Para cada ítem dentro del límite dado por @:
    for i in range(1, arroba+1):
        item = recomendaciones[i-1]
        # Si el item está en el top_n_verdadero, es relevante.
        if item in top_n_verdadero[0:limite_top_n_verdadero]:
            relevante = 1
        # Si no, no es relevante:
        else:
            relevante = 0
        # Sumamos el gain descontado:
        dcg += (2**relevante - 1) / log2(i + 1)
    # Retornamos el resultado final:
    return dcg


def distancia(item_1, item_2):
    return scipy.spatial.distance.cosine(item_1, item_2)


def obtener_access_token_para_la_api_de_spotify(client_id, client_secret):
    client_id_and_client_secret = client_id + ':' + client_secret
    client_id_and_client_secret = str(b64encode(client_id_and_client_secret.encode()))[2:-1]
    response = requests.post(
        url='https://accounts.spotify.com/api/token',
        headers={'Authorization': 'Basic ' + client_id_and_client_secret},
        data={'grant_type': 'client_credentials'},
        json=True
    )
    return json.loads(response.text)['access_token']


def probabilidad_de_conocer_dado_item(item, interacciones, usuarios):
    usuarios_que_conocen_el_item = interacciones[interacciones['track_id_clean'] == item]['session_id'].unique()
    return len(usuarios_que_conocen_el_item) / len(usuarios)


def probabilidad_de_item(item, interacciones):
    return len(interacciones[interacciones['track_id_clean'] == item]) / len(interacciones)


def probabilidad_de_item_dada_lista_de_recomendaciones(item, recomendaciones, items, interacciones):
    return probabilidad_de_item(item, interacciones) * len(items) / len(recomendaciones)


def configurar_pyplot():
    plt.figure(figsize=(16, 9))
    plt.xlabel('Cantidad de reproducciones', fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.ylabel('Cantidad de canciones', fontsize='x-large')
    plt.yticks(fontsize='large')
