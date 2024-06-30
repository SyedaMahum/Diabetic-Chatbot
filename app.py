from flask import Flask, render_template, request
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
from nltk.tokenize import sent_tokenize
import pickle
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load pretrained models and tokenizers
with open(r'sample_data/diabeticBot_model.pkl', 'rb') as f:
    diabeticBot = pickle.load(f)

with open(r'sample_data/bert_tokenizer.pkl', 'rb') as f:
    tokenizer2 = pickle.load(f)

with open(r'sample_data/bert_model.pkl', 'rb') as f:
    model2 = pickle.load(f)

# Read the CSV file with specified encoding
df = pd.read_csv(r"sample_data/questions (1).csv", encoding='latin1')

# Tokenize the answers into sentences
all_answers = ' '.join(df['Answers'])
data = sent_tokenize(all_answers)

# Load precomputed embeddings
with open("sample_data/Embeddings2.pkl", 'rb') as f:
    embeddings = pickle.load(f)

# Hardcoded greetings
GREETINGS = {
    "hi": "Hi! How can I help you?",
    "hello": "Hello! How can I help you?"
}

# List to store previous questions
previous_questions = []

@app.route('/')
def index():
    return render_template('index.html', previous_questions=previous_questions)

@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        # Get user input question from HTML form
        user_question = request.form['question']
        
        # Add user question to previous questions list
        previous_questions.append(user_question)
        
        # Check for hardcoded greetings
        user_question_lower = user_question.lower()
        if user_question_lower in GREETINGS:
            answer_text = GREETINGS[user_question_lower]
            context = ""  # No context for greetings
        else:
            # Tokenize and encode the user question
            example_encoding = tokenizer2.batch_encode_plus(
                [user_question],
                padding=True,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            example_input_ids = example_encoding['input_ids']
            example_attention_mask = example_encoding['attention_mask']

            # Generate embeddings for the user question
            with torch.no_grad():
                example_outputs = model2(example_input_ids, attention_mask=example_attention_mask)
                example_sentence_embedding = example_outputs.last_hidden_state.mean(dim=1)

            # Compute cosine similarity between the original sentence embeddings and the user question embedding
            similarity_score = cosine_similarity(embeddings, example_sentence_embedding)

            # Get top 10 similar sentences
            top_indices = np.argsort(similarity_score.squeeze())[::-1][:5]
            top_sentences = [data[idx] for idx in top_indices]

            # Combine the top sentences into a single context
            context = " ".join(top_sentences)

            # Use the diabeticBot pipeline to answer the user question
            answer = diabeticBot({
                "question": user_question,
                "context": context
            })

            answer_text = answer['answer']
        
        print("Context:", context)  # Debug print statement
        return render_template('index.html', question=user_question, answer=answer_text, context=context)

if __name__ == '__main__':
    app.run(debug=True)
