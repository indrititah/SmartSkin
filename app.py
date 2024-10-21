from process import preparation, generate_response
from flask import Flask, render_template, request, jsonify
# download nltk
# download nltk




# ...
preparation()
#Start Chatbot
app = Flask(__name__)

# Panggil preparation() di sini


@app.route("/")
def index():
    return render_template("index.html")
@app.route('/askhere')
def ask_here():
    return render_template('askhere.html')

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

if __name__ == "__main__":
    app.run(debug=True)