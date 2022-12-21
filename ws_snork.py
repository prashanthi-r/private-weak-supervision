import logging
import torch
import numpy as np
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import MLPModel
from wrench.labelmodel import MajorityVoting
from privacy.research.pate_2017 import analysis # install tensorflow's privacy repository in the root to import this without error
# i have uploaded just the pate_2017 files to our root directory with the change we made for the project
# because the privacy repository is too huge.
from privacy.research.pate_2017 import aggregation
from wrench.dataset.utils import check_weak_labels
from snorkel.utils import probs_to_preds
import time
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

#### Load dataset
dataset_home = 'wrench/datasets(1)/datasets'
data = 'census'


#### Extract data features using pre-trained BERT model and cache it
extract_fn = 'bert'
model_name = 'bert-base-cased'
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=True, extract_fn=extract_fn,
                                                 cache_name=extract_fn, model_name=model_name)
PATE = 1 # 0 by default (just snorkel), 1 if snorkel+pate
# EPS = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
# EPS = [0.0001, 0.001]
EPS = [1.33]

eps_dict ={}
#### Generate soft training label via a label model
#### The weak labels provided by supervision sources are alreadly encoded in dataset object
L = np.array(check_weak_labels(train_data))
# L_T = np.transpose(L)
# with open('adult_l_weak_label.npy', 'wb') as f:
    # np.save(f,L_T)

device = torch.device('cuda:0')
n_steps = 100000
batch_size = 128
test_batch_size = 1000
patience = 200
evaluation_step = 50
target='acc'

if not PATE:
    label_model = MajorityVoting()
    label_model.fit(train_data)
    soft_label = label_model.predict_proba(train_data)
    # print(soft_label.shape)
    hard_label = probs_to_preds(soft_label)
    print("\nSnorkel Stats: \n")
    # data_dep_eps, data_ind_eps = analysis.perform_analysis(teacher_preds = L, indices = hard_label, noise_eps=EPS, num_classes=train_data.n_class)
    # print("Data Independent Epsilon:", data_ind_eps)
    # print("Data Dependent Epsilon:", data_dep_eps)
    model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
    model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label,
              device=device, metric=target, patience=patience, evaluation_step=evaluation_step)

else:
    for eps in EPS:
        print("\nSnorkel+PATE Stats: \n")
        # print(L[:10])
        # print(np.array(L).shape)
        # print(L.shape)
        stdnt_labels = aggregation.noisy_max(logits=L, lap_scale=1/eps, return_clean_votes=False, num_classes=train_data.n_class)

        # print("stdnt_labels: ", stdnt_labels[:100])
        # print("stdnt_labels shape: ", stdnt_labels.shape)

        data_dep_eps, data_ind_eps = analysis.perform_analysis(teacher_preds = L, indices = stdnt_labels, noise_eps=eps, num_classes=train_data.n_class)
        # time.sleep(10)
        model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
        hist = model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=stdnt_labels,
                  device=device, metric=target, patience=patience, evaluation_step=evaluation_step)
        hist_keys = list(hist.keys())
        eps_dict[eps] = [hist[hist_keys[-1]],data_ind_eps,data_dep_eps]
        print("Data Independent Epsilon:", data_ind_eps)
        print("Data Dependent Epsilon:", data_dep_eps)
print(eps_dict)

# print(hard_label.shape)
# print("hard_labels: ", hard_label[:100])

# Added by Prashanthi

#### Train a MLP classifier with soft label
# device = torch.device('cuda:0')
# n_steps = 100000
# batch_size = 128
# test_batch_size = 1000
# patience = 200
# evaluation_step = 50
# target='acc'
#
# model = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
# history = model.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label,
#                     device=device, metric=target, patience=patience, evaluation_step=evaluation_step)
#
# #### Evaluate the trained model
# metric_value = model.test(test_data, target)

### We can also train a MLP classifier with hard label

# hard_label = probs_to_preds(soft_label)
# model1 = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
# model1.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=hard_label,
#           device=device, metric=target, patience=patience, evaluation_step=evaluation_step)

# Training an MLP with Snorkel+PATE
# model2 = MLPModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)
# model2.fit(dataset_train=train_data, dataset_valid=valid_data, y_train=stdnt_labels,
#           device=device, metric=target, patience=patience, evaluation_step=evaluation_step)
