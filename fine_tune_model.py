import torch
import torch.nn as nn
from collections import OrderedDict
from skorch import NeuralNet
from sklearn.pipeline import Pipeline
import numpy as np

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

def freeze_model(model, freeze_early_layers=True):
    """
    Freeze model parameters selectively.
    If freeze_early_layers is True, only freeze early layers.
    """
    for name, param in model.named_parameters():
        if freeze_early_layers:
            # Freeze only early layers (conv1, conv2)
            if any(x in name for x in ['conv1', 'conv2']):
                param.requires_grad = False
        else:
            # Freeze all layers
            param.requires_grad = False
    return model

class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten
        return self.fc(x)

class FullModel(nn.Module):
    def __init__(self, embedding, num_classes):
        super().__init__()
        self.embedding = embedding
        # Calculate input size for classification head
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 385, dtype=torch.float32)  # Explicitly use float32
            dummy_output = self.embedding(dummy_input)
            input_size = dummy_output.numel()
        self.classifier = ClassificationHead(input_size, num_classes)
        
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        x = self.embedding(x)
        return self.classifier(x)

class FineTunedNeuralNet(NeuralNet):
    def __init__(
            self,
            *args,
            criterion=nn.CrossEntropyLoss,
            unique_name=None,
            **kwargs
    ):
        super().__init__(
            *args,
            criterion=criterion,
            **kwargs
        )
        self.initialize()
        self.unique_name = unique_name

    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true)

    def __repr__(self):
        return super().__repr__() + self.unique_name

def create_fine_tuned_pipeline(torch_module, num_classes=3):
    """
    Create a fine-tuned pipeline from a pre-trained model.
    
    Args:
        torch_module: The pre-trained model
        num_classes: Number of output classes
        
    Returns:
        sklearn Pipeline with the fine-tuned model
    """
    # Remove classification layers and prepare embedding
    embedding = remove_clf_layers(torch_module).float()  # Convert to float32
    
    # Create the full model
    full_model = FullModel(embedding, num_classes=num_classes)
    full_model = freeze_model(full_model, freeze_early_layers=True)
    
    # Create the pipeline
    pipeline = Pipeline([
        ('model', FineTunedNeuralNet(
            full_model,
            criterion=nn.CrossEntropyLoss,
            max_epochs=50,
            batch_size=32,
            optimizer=torch.optim.Adam,
            optimizer__lr=0.001,
            unique_name='fine_tuned_model'
        )),
    ])
    
    return pipeline 