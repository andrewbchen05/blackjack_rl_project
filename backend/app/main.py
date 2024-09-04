from flask import Flask
from routes import game_routes

app = Flask(__name__)

app.register_blueprint(game_routes)

if __name__ == '__main__':
    app.run(debug=True)