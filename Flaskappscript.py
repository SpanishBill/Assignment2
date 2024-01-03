import os
print ("Current Working Directory:", os.getcwd())
print("Files in the current directory:", os.listdir())

from flask import Flask, render_template, request
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("Assignment2_FINAL.model")

@app.route('/', methods=['GET', 'POST'])
def antonym_tool ():
    antonym = []
    if request.method == "POST":
        word = request.form.get("word").lower() #convert to lowercase
        if word:
            try:
                # Fetch the top similar word
                antonym = model.wv.most_similar(word, topn=1)[0][0]
            except KeyError:
                antonym = ["Word not in model's vocabulary."]
    return render_template ('antonym_finder.html', antonym=antonym)

if __name__ == '__main__':
     app.run(debug=True)