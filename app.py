from flask import Flask
from flask_cors import CORS
from webhook.routes import messenger_bp
from config import LOG_FILE
import logging

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Register Blueprint
app.register_blueprint(messenger_bp)

@app.route('/')
def home():
    return "Facebook Messenger Chatbot Webhook is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1337)
