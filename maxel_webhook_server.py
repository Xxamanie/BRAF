from flask import Flask, request, jsonify
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Maxel secret from environment
MAXEL_SECRET = os.environ.get("MAXEL_SECRET", "default_secret")

@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle Maxel webhook notifications"""
    try:
        data = request.json
        
        # Simple secret check for authentication
        if request.headers.get("X-Maxel-Secret") != MAXEL_SECRET:
            logger.warning(f"Unauthorized webhook attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Log webhook payload
        timestamp = datetime.now().isoformat()
        logger.info(f"Webhook received at {timestamp}: {data}")
        
        # Handle different webhook event types
        event_type = data.get("event_type", "unknown")
        
        if event_type == "payment_received":
            handle_payment_received(data)
        elif event_type == "payment_failed":
            handle_payment_failed(data)
        elif event_type == "withdrawal_completed":
            handle_withdrawal_completed(data)
        else:
            logger.info(f"Unknown event type: {event_type}")
        
        return jsonify({"status": "success", "timestamp": timestamp}), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def handle_payment_received(data):
    """Process payment received notification"""
    payment_id = data.get("payment_id")
    amount = data.get("amount")
    currency = data.get("currency")
    
    logger.info(f"Payment received: ID={payment_id}, Amount={amount} {currency}")
    
    # Add your payment processing logic here
    # For example: update database, send confirmation email, etc.

def handle_payment_failed(data):
    """Process payment failed notification"""
    payment_id = data.get("payment_id")
    reason = data.get("reason")
    
    logger.warning(f"Payment failed: ID={payment_id}, Reason={reason}")
    
    # Add your failure handling logic here
    # For example: notify user, retry payment, etc.

def handle_withdrawal_completed(data):
    """Process withdrawal completed notification"""
    withdrawal_id = data.get("withdrawal_id")
    amount = data.get("amount")
    currency = data.get("currency")
    
    logger.info(f"Withdrawal completed: ID={withdrawal_id}, Amount={amount} {currency}")
    
    # Add your withdrawal processing logic here
    # For example: update user balance, send notification, etc.

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route("/", methods=["GET"])
def index():
    """Basic info endpoint"""
    return jsonify({
        "service": "Maxel Webhook Server",
        "status": "running",
        "endpoints": {
            "webhook": "/webhook (POST)",
            "health": "/health (GET)"
        }
    }), 200

if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Maxel webhook server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)