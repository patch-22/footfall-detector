import requests

API_BASE_URL = 'http://172.16.0.174:3000/events'

def report_entrance():
    body = {
        'count': 1,
        'target': 'pivotal'
    }

    # requests.post(API_BASE_URL, json=body)



def report_exit():
    body = {
        'count': -1,
        'target': 'pivotal'
    }

    # requests.post(API_BASE_URL, json=body)