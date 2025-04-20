from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the chatbot model with text-generation task
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    
    # Generate a response from the model
    bot_response = chatbot(user_input, max_length=50, num_return_sequences=1)
    return bot_response[0]['generated_text']

if __name__ == '__main__':
    app.run(debug=True)
