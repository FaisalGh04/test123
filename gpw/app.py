from flask import Flask, render_template, request, jsonify, Response, g
from openai import OpenAI
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
logger.debug(f"Template folder path: {app.template_folder}")

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("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"))  # Make sure to set your OpenAI API key in the environment variables

# Template for the prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Store conversation context in Flask's g object
@app.before_request
def initialize_context():
    if not hasattr(g, 'context'):
        g.context = ""
    if not hasattr(g, 'MAX_CONTEXT_LENGTH'):
        g.MAX_CONTEXT_LENGTH = 500  # Reduce context size
    logger.debug(f"Context initialized: {g.context}")

# Routes
@app.route("/")
def home():
    logger.debug("Rendering home page...")
    return render_template("index.html")

@app.route("/about")
def about():
    logger.debug("Rendering about page...")
    return render_template("about.html")

@app.route("/services")
def services():
    logger.debug("Rendering services page...")
    return render_template("services.html")

@app.route("/contact")
def contact():
    logger.debug("Rendering contact page...")
    return render_template("contact.html")

@app.route("/chat", methods=["GET"])
def chat_stream():
    user_input = request.args.get("message", "").strip()
    logger.debug(f"Received user input: {user_input}")

    # Handle exit command
    if user_input.lower() == "exit":
        logger.debug("Exit command received.")
        return jsonify({"response": "Goodbye!"})

    # Limit the context size to avoid excessive memory usage
    if len(g.context) > g.MAX_CONTEXT_LENGTH:
        g.context = g.context[-g.MAX_CONTEXT_LENGTH:]
        logger.debug(f"Context trimmed to: {g.context}")

    try:
        logger.debug("Sending request to GPT-4 API...")
        response = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-4" or "gpt-4-1106-preview" depending on your access
            messages=[
                {"role": "system", "content": template.format(context=g.context, question=user_input)},
                {"role": "user", "content": user_input}
            ],
            stream=True  # Enable streaming
        )

        # Stream the response back to the client
        def generate():
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
                    time.sleep(0.1)  # Simulate a delay between words
            yield "data: [END]\n\n"  # Signal the end of the response

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's port or default to 5000
    logger.debug(f"Starting Flask app on port {port}...")
    app.run(debug=False, host="0.0.0.0", port=port)