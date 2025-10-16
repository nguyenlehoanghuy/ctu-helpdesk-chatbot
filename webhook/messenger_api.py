import requests
from config import PAGE_ACCESS_TOKEN

def send_message(recipient_id, text):
    url = 'https://graph.facebook.com/v19.0/me/messages'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'recipient': {'id': recipient_id},
        'message': {'text': text},
        'messaging_type': 'RESPONSE'
    }
    params = {'access_token': PAGE_ACCESS_TOKEN}

    res = requests.post(url, headers=headers, params=params, json=payload)
    if res.status_code != 200:
        print("❌ Error sending message:", res.status_code, res.text)
    else:
        print("✅ Sent:", res.json())
