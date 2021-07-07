from flask import Flask, render_template, request
from utils import text_classifier

'''    only for demo purpose utlis.py contains the functions    '''

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/process', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        query = request.form['query']
        classified = text_classifier(query)

    return render_template('classifier.html', classified=classified)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
