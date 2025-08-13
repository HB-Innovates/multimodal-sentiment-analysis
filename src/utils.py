import torch

def save_model(model, filepath):
    """Save the trained PyTorch model to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(filepath, model_class=None):
    """Load a PyTorch model from the specified filepath. You must provide the model class."""
    if model_class is None:
        raise ValueError("model_class must be provided to load the model.")
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

def plot_metrics(metrics):
    """Plot training and evaluation metrics."""
    import matplotlib.pyplot as plt

    for metric, values in metrics.items():
        plt.plot(values, label=metric)

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Metrics over Epochs')
    plt.legend()
    plt.show()