# %%
from collections import OrderedDict
from torch import nn
from skorch import NeuralNet
from skorch.utils import to_numpy
from sklearn.base import TransformerMixin
from braindecode.models import EEGNetv4
from huggingface_hub import hf_hub_download
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from moabb.paradigms import MotorImagery
from moabb.datasets import Weibo2014
from moabb.evaluations import WithinSessionEvaluation

import torch
import pandas as pd
import pickle

# %%
def remove_clf_layers(model: nn.Sequential):
    """
    Remove the classification layers from braindecode models.
    Tested on EEGNetv4, Deep4Net (i.e. DeepConvNet), and EEGResNet.
    """
    new_layers = []
    for name, layer in model.named_children():
        if 'classif' in name:
            continue
        if 'softmax' in name:
            continue
        new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# %%
class FrozenNeuralNetTransformer(NeuralNet, TransformerMixin):
    def __init__(
            self,
            *args,
            criterion=nn.MSELoss,  # should be unused
            unique_name=None,  # needed for a unique digest in MOABB
            **kwargs
    ):
        super().__init__(
            *args,
            criterion=criterion,
            **kwargs
        )
        self.initialize()
        self.unique_name = unique_name

    def fit(self, X, y=None, **fit_params):
        return self  # do nothing

    def transform(self, X):
        X = self.infer(X)
        return to_numpy(X)

    def __repr__(self):
        return super().__repr__() + self.unique_name
    
def flatten_batched(X):
    return X.reshape(X.shape[0], -1)

# %%
path_kwargs = hf_hub_download(
    repo_id='PierreGtch/EEGNetv4',
    filename='EEGNetv4_Lee2019_MI/kwargs.pkl',
)
path_params = hf_hub_download(
    repo_id='PierreGtch/EEGNetv4',
    filename='EEGNetv4_Lee2019_MI/model-params.pkl',
)
with open(path_kwargs, 'rb') as f:
    kwargs = pickle.load(f)
module_cls = kwargs['module_cls']
module_kwargs = kwargs['module_kwargs']

torch_module = module_cls(**module_kwargs)
torch_module.load_state_dict(torch.load(path_params, map_location='cpu'))
embedding = freeze_model(remove_clf_layers(torch_module)).double()

sklearn_pipeline = Pipeline([
    ('embedding', FrozenNeuralNetTransformer(embedding, unique_name='pretrained_Lee2019')),
    ('flatten', FunctionTransformer(flatten_batched)),
    ('classifier', KNeighborsClassifier(n_neighbors=7)),
])

# %%
paradigm = MotorImagery(
    channels=['C3', 'Cz', 'C4'],  # Same as the ones used to pre-train the embedding
    events=['left_hand', 'right_hand'],
    n_classes=2,
    fmin=0.5,
    fmax=40,
    tmin=0,
    tmax=3,
    resample=128
)
datasets = [Weibo2014()]
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
    suffix='demo',
)

# %%
results = evaluation.process(pipelines=dict(demo_pipeline=sklearn_pipeline))

# %%
data = {
    "accuracy": [results['score'].mean()],
    "f1": [results["f1"].mean()],
    "recall": [results["recall"].mean()],
    "specificity": [results["specificity"].mean()],
    "precision": [results["precision"].mean()]     
    } 
df = pd.DataFrame(data)
print(df)


