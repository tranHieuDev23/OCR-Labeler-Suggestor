from flask import Flask, jsonify, request
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import json

config = Cfg.load_config_from_file('config/vgg-seq2seq.yml')
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
        'error': 'Error: ' + str(error)
    }
    return query_result


def parseRegionJson(jsonStr):
    jsonObj = json.loads(jsonStr)
    regions = []
    for item in jsonObj['regions']:
        newRegion = []
        for coord in item:
            newRegion.append([coord])
        regions.append(newRegion)
    return regions


@app.route('/query', methods=['POST'])
def queryimg():
    imgData = request.files['file']
    regionJson = request.form['regionJson']
    regions = parseRegionJson(regionJson)
    try:
        img = Image.open(imgData)
        start = time.time()
        label_predicts = model.predict_with_boxes(img, regions)
        time_pred = str(time.time() - start)
        result = {'result': label_predicts, 'predict time': time_pred}
        return jsonify_str(result)
    except Exception as ex:
        return jsonify(create_error_result(ex))


if __name__ == '__main__':
    app.run('localhost', os.getenv('SUGGESTOR_PORT'),
            threaded=True, debug=True)
