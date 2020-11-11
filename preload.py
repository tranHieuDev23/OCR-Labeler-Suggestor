from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_file('config/vgg-seq2seq.yml')
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False
model = Predictor(config)
