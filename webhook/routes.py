from flask import Blueprint, request, jsonify
from config import VERIFY_TOKEN
from webhook.handlers import handle_user_message
from webhook.messenger_api import send_message
import json

messenger_bp = Blueprint('messenger_bp', __name__)

# Verification endpoint
@messenger_bp.route('/webhook', methods=['GET'])
def verify():
    token_sent = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if token_sent == VERIFY_TOKEN:
        return challenge
    return "Invalid verification token", 403


# Handle incoming message
'''
@messenger_bp.route('/webhook', methods=['POST'])
def receive_message():
    body = request.get_json()
    print("📩 Incoming:", json.dumps(body, indent=2, ensure_ascii=False))
    if body.get('object') == 'page':
        for entry in body.get('entry', []):
            for event in entry.get('messaging', []):
                if 'message' in event and 'text' in event['message']:
                    sender_id = event['sender']['id']
                    user_message = event['message']['text']
                    # reply = handle_user_message(user_message)
                    send_message(sender_id, "Xin chào!")
                else:
                    print("⚠️ Event không phải message:", event)
    return "Message processed", 200
'''
processed_messages = set()  # Ở môi trường thực nên dùng DB hoặc Redis

@messenger_bp.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    if data["object"] == "page":
        for entry in data["entry"]:
            for event in entry["messaging"]:
                if "message" in event and "mid" in event["message"]:
                    message_id = event["message"]["mid"]
                    if message_id in processed_messages:
                        print(f"⚠️ Tin nhắn {message_id} đã xử lý, bỏ qua")
                        continue
                    processed_messages.add(message_id)

                    sender_id = event["sender"]["id"]
                    message = event["message"]["text"]
                    reply = handle_user_message(message)
                    send_message(sender_id, reply)
    return "ok", 200

