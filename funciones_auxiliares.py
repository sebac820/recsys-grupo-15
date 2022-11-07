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


def configurar_pyplot(xlabel='', ylabel=''):
    plt.figure(figsize=(16, 9))
    plt.xlabel(xlabel, fontsize='x-large')
    plt.xticks(fontsize='large')
    plt.ylabel(ylabel, fontsize='x-large')
    plt.yticks(fontsize='large')


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


def probabilidad_de_conocer_item(item, interacciones):
    numero_de_usuarios_que_conocen_el_item = len(interacciones[interacciones['track_id'] == item]['session_id'].unique())
    numero_de_usuarios = len(interacciones['session_id'].unique())
    return numero_de_usuarios_que_conocen_el_item / numero_de_usuarios


def probabilidad_de_item_1_dado_item_2(item_1, item_2, interacciones):
    usuarios_que_han_accedido_a_item_1 = interacciones[interacciones['track_id'] == item_1]['session_id'].unique()
    usuarios_que_han_accedido_a_item_2 = interacciones[interacciones['track_id'] == item_2]['session_id'].unique()
    numero_de_usuarios_que_han_accedido_a_ambos_items = 0
    for usuario in usuarios_que_han_accedido_a_item_1:
        if usuario in usuarios_que_han_accedido_a_item_2:
            numero_de_usuarios_que_han_accedido_a_ambos_items += 1
    if len(usuarios_que_han_accedido_a_item_2) > 0:
        return numero_de_usuarios_que_han_accedido_a_ambos_items / len(usuarios_que_han_accedido_a_item_2)
    return 0


def probabilidad_de_item_dado_usuario(item, usuario, interacciones):
    numero_de_interacciones_con_item = len(interacciones[(interacciones['session_id'] == usuario) & (interacciones['track_id'] == item)])
    numero_de_interacciones = len(interacciones[interacciones['session_id'] == usuario])
    probabilidad_de_item_dado_usuario = numero_de_interacciones_con_item / numero_de_interacciones
    if probabilidad_de_item_dado_usuario > 0:
        return probabilidad_de_item_dado_usuario
    return 0.0000000001


def probabilidad_de_que_item_sea_conocido_por_usuario(item, usuario, interacciones):
    probabilidades = 0
    perfil_de_usuario = interacciones[interacciones['session_id'] == usuario]['track_id'].unique()
    for otro_item in perfil_de_usuario:
        probabilidades += (
            probabilidad_de_conocer_item(otro_item, interacciones) *
            probabilidad_de_item_1_dado_item_2(item, otro_item, interacciones) *
            probabilidad_de_item_dado_usuario(otro_item, usuario, interacciones) /
            probabilidad_de_item_dado_usuario(item, usuario, interacciones)
        )
    return probabilidades
