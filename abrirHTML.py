import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

x = [0] * 3000

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def sessions():
    return render_template('index.html')

@socketio.on('acquire')
def start_acquisition(json):
    emit('serverResponse', {'data': 0, 'signal': x})

if __name__ == '__main__':
    socketio.run(app, debug = True)