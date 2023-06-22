import requests

headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'amount': '10.00',
    'description': 'this is payload',
}

response = requests.post(
    'https://example.com/v1/postendpoint',
    headers=headers,
    json=json_data,
    auth=('USER', 'PASSWORD'),
)
