import os
from flask import Flask, jsonify, request, Response

import json
from prediction import predict
import numpy as np
from keras.models import load_model
import gc

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def flask_app():
    app = Flask(__name__)
    model = load_model('models/weightsMulti.hdf5')
    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up'

    @app.route('/inference', methods=['POST'])
    def start():
        file = request.files['file']
        pred = predict(file, model)
        return Response(json.dumps(pred, cls=MyEncoder), mimetype="application/json")
        # return jsonify({"prediction:":pred})
    gc.collect()
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')