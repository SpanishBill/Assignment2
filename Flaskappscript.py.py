from flask import Flask, render_template_string, request
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the trained Word2Vec model
model = Word2Vec.load("Assignment2.model")

html_form = '''
<!DOCTYPE html>
<html>
<head>
    <title>Beta Antonym Finder</title>
</head>
<body>
    <form method="post" action="/">
        <input type="text" name="word" id="word" placeholder="Enter a word">
        <input type="submit" value="Get Antonym">
    </form>
    {% if antonym %}
        <p>Antonym: {{ antonym }}</p>
    {% elif antonym is not None %}
        <p>Word not found in the model.</p>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    antonym = None
    if request.method == "POST":
        word = request.form.get("word")
        if word:
            try:
                # Fetch the top similar word
                antonym = model.wv.most_similar(word, topn=1)[0][0]
            except KeyError:
                # Word not in model's vocabulary
                antonym = None
    return render_template_string(html_form, antonym=antonym)

if __name__ == '__main__':
    app.run(debug=True)