from flask import Flask, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

qa_pairs = {
    "What is Python?": "Python is a high-level programming language known for its simplicity and readability.",
    "What is JavaScript?": "JavaScript is a scripting language commonly used for web development.",
    "What is Java?": "Java is a widely-used programming language known for its platform independence.",
    "Tell me able Python?": "Python is a high-level programming language known for its simplicity and readability.",
}

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(list(qa_pairs.keys()))
@app.route('/chat', methods=['POST'])
def ask():
    question = request.json['question']
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, question_embeddings)
    print(similarities)
    # similarities.sort(True, float)
    # print(similarities.remove(np.argmax(similarities)))
    # print(similarities.remove(np.argmax(similarities)))
    # print(similarities.remove(np.argmax(similarities)))
    
    partition = np.argpartition(similarities.ravel(), -3)[-3:]
    print(partition)
    max_three = similarities[partition//8,partition%8]
    print(max_three)
    most_similar_index = np.argmax(similarities)
    print(most_similar_index)

    # answer = list(qa_pairs.values())[partition[0]]
    # print(answer);
    # answer = list(qa_pairs.values())[partition[1]]
    # print(answer);
    # answer = list(qa_pairs.values())[partition[2]]
    # print(answer);
    answer = list(qa_pairs.values())[most_similar_index]
    return {'answer': answer}

if __name__ == '__main__':
    app.run(debug=True)


