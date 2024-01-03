from flask import Flask, render_template, request
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("word2vec.model")

@app.route("/", methods=["GET", "POST"])
def index():
    similar_words = []

    if request.method == "POST":
        word = request.form["word"]
        if word:
            try:
                # Get the most similar words
                similar_words = model.wv.most_similar(word, topn=5)
            except KeyError:
                similar_words = ["Word not found in the model."]

    return render_template("index.html", similar_words=similar_words)

if __name__ == "__main__":
    app.run(debug=True)