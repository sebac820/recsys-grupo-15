from base64 import b64encode
import json
import requests


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
