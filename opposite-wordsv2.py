from flask import Flask, render_template, request
from gensim.models import Word2Vec

app = Flask(__name__)

model = Word2Vec.load("word2vec.model")

@app.route("/", methods=["GET", "POST"])
def index():
    opposite_words = []

    if request.method == "POST":
        word = request.form["word"]
        if word:
            try:
                reference_pair = ("old", "young")
                result_vector = model.wv[word] - model.wv[reference_pair[0]] + model.wv[reference_pair[1]]
                opposite_words = model.wv.similar_by_vector(result_vector)
            except KeyError:
                opposite_words = ["Word not found in the model"]

    return render_template("index.html", opposite_words=opposite_words)

if __name__ == "__main__":
    app.run(debug=True)
