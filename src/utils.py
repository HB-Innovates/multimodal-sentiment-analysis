def save_model(model, filepath):
    """Save the trained model to the specified filepath."""
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load a model from the specified filepath."""
    import joblib
    return joblib.load(filepath)

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