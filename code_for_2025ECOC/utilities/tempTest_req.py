import requests

headers = {
    'Origin': 'http://192.168.1.149',
    'Referer': 'http://192.168.1.149/io_cgi.ssi',
}

data = {
    'OPOUT': '16.4',
    'Display': 'Submit',
}

res1 = requests.get('http://192.168.1.149/io_http.ssi')
res2 = requests.get('http://192.168.1.149/io_cgi.ssi')

response = requests.post('http://192.168.1.149/config1.cgi', headers=headers, data=data, verify=False)

print(response.status_code)
print(response.text)