from flask import Flask, render_template, request, jsonify, Response, g
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")
logger.debug(f"Template folder path: {app.template_folder}")

# Initialize the model and prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

try:
    logger.debug("Initializing OllamaLLM model...")
    model = OllamaLLM(model="llama3:latest")  # Ensure the model name is correct
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    logger.debug("Model and chain initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing model or chain: {e}")
    model = None
    chain = None

# Store conversation context in Flask's g object
@app.before_request
def initialize_context():
    if not hasattr(g, 'context'):
        g.context = ""
    if not hasattr(g, 'MAX_CONTEXT_LENGTH'):
        g.MAX_CONTEXT_LENGTH = 500  # Reduce context size
    logger.debug(f"Context initialized: {g.context}")

# Warm up the model
if chain:
    try:
        logger.debug("Warming up the model...")
        chain.invoke({"context": "", "question": "Hello"})
        logger.debug("Model warmed up successfully.")
    except Exception as e:
        logger.error(f"Error warming up the model: {e}")

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
    if not chain:
        logger.error("Chain not initialized. Model may not be available.")
        return jsonify({"response": "Error: Model not initialized"}), 500

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
        logger.debug("Invoking the chain...")
        result = chain.invoke({"context": g.context, "question": user_input})
        g.context += f"\nUser: {user_input}\nAI: {result}"
        logger.debug(f"Generated response: {result}")

        # Stream the response back to the client
        def generate():
            for word in result.split():
                yield f"data: {word}\n\n"
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