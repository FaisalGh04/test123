from flask import Flask, render_template, request, jsonify, Response, g
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import os

app = Flask(__name__, template_folder="templates")  # Use a relative path for templates
print(f"Template folder path: {app.template_folder}")

# Initialize the model and prompt
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""
try:
    model = OllamaLLM(model="llama2:latest")  # Use the correct model name
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
except Exception as e:
    print(f"Error initializing model or chain: {e}")
    model = None
    chain = None

# Store conversation context in Flask's g object
@app.before_request
def initialize_context():
    if not hasattr(g, 'context'):
        g.context = ""
    if not hasattr(g, 'MAX_CONTEXT_LENGTH'):
        g.MAX_CONTEXT_LENGTH = 500  # Reduce context size

# Warm up the model
if chain:
    try:
        chain.invoke({"context": "", "question": "Hello"})
    except Exception as e:
        print(f"Error warming up the model: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not chain:
        return jsonify({"response": "Error: Model not initialized"}), 500

    user_input = request.json.get("message", "").strip()

    # Handle exit command
    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})

    # Limit the context size to avoid excessive memory usage
    if len(g.context) > g.MAX_CONTEXT_LENGTH:
        g.context = g.context[-g.MAX_CONTEXT_LENGTH:]

    try:
        # Invoke the chain
        result = chain.invoke({"context": g.context, "question": user_input})
        g.context += f"\nUser: {user_input}\nAI: {result}"

        # Return the text response in chunks
        def generate():
            for word in result.split():
                yield f"data: {word}\n\n"
                time.sleep(0.1)  # Simulate a delay between words

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku's port or default to 5000
    app.run(debug=True, host="0.0.0.0", port=port)