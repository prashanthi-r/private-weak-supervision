import logging
import torch
import numpy as np
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import MLPModel
from wrench.labelmodel import MajorityVoting
from privacy.research.pate_2017 import analysis
from privacy.research.pate_2017 import aggregation
from wrench.dataset.utils import check_weak_labels
from snorkel.utils import probs_to_preds

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset
dataset_home = 'wrench/datasets(1)/datasets'
data = 'census'
EPS = 1.0

#### Extract data features using pre-trained BERT model and cache it
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)
print(train_data.__dict__.keys())
print(train_data.n_class)                                                
