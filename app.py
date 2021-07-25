from flask import Flask, request, render_template
from werkzeug.exceptions import Forbidden, HTTPException, NotFound, RequestTimeout, Unauthorized
import os
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(NotFound)
def page_not_found_handler(e: HTTPException):
    return '<h1>404.html</h1>', 404


@app.errorhandler(Unauthorized)
def unauthorized_handler(e: HTTPException):
    return '<h1>401.html</h1>', 401


@app.errorhandler(Forbidden)
def forbidden_handler(e: HTTPException):
    return '<h1>403.html</h1>', 403


@app.errorhandler(RequestTimeout)
def request_timeout_handler(e: HTTPException):
    return '<h1>408.html</h1>', 408


if __name__ == '__main__':
    os.environ.setdefault('Flask_SETTINGS_MODULE', 'helloworld.settings')
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    port = int(os.environ.get("PORT", 33507))
    app.run(debug=True)
