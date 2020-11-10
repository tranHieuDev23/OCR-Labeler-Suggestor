from flask import Flask, jsonify, request
from io import BytesIO
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time


config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
model = Predictor(config)


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


app = Flask(__name__)


def create_error_result(error=None):
    query_result = {
        'results': 'Error: ' + str(error)
    }
    return query_result


@app.route("/query", methods=['GET', 'POST'])
def queryimg():
    result = None
    if request.method == "POST":
        data = request.get_data()
        try:
            img = Image.open(BytesIO(data))
            start = time.time()
            result_text = model.predict(img)
            time_pred = str(time.time() - start)
            result = {"result: ": result_text, "predict time": time_pred}
            return jsonify_str(result)
        except Exception as ex:
            return jsonify(create_error_result(ex))


if __name__ == "__main__":
    app.run("localhost", 1912, threaded=True, debug=False)
