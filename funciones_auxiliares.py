from base64 import b64encode
from math import log2
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
