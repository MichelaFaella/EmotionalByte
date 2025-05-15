import matplotlib.pyplot as plt
import os

def plotLoss(logs, epochs):

    epochs = list(range(1, epochs + 1))

    # Cross-Entropy and Task Loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, logs['loss_task'], label='loss_task')
    plt.plot(epochs, logs['loss_ce_t'], label='loss_ce_text')
    plt.plot(epochs, logs['loss_ce_a'], label='loss_ce_audio')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss_task and loss_ce losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ShowOrSavePlot("plot", "loss_ce")

    # KL Divergence Losses
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, logs['loss_kl_t'], label='loss_kl_text')
    plt.plot(epochs, logs['loss_kl_a'], label='loss_kl_audio')
    plt.xlabel("Epochs")
    plt.ylabel("KL Loss")
    plt.title("loss_KL losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ShowOrSavePlot("plot", "loss_kl")

    # total loss  
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, logs['loss'], label='loss')
    plt.xlabel("Epochs")
    plt.ylabel("total Loss")
    plt.title("Total loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ShowOrSavePlot("plot", "loss")

def ShowOrSavePlot(path=None, filename=None):
    """
    Displays a plot or saves it to a specified path.

    :param path: Directory path to save the plot. If None, the plot is displayed.
    :param filename: Name of the file (without extension) to save the plot. Defaults to 'img' if not specified.
    """
    if path is None or path == '':
        plt.show()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None or filename == '':
            filename = 'img'
        plt.savefig(f"{path}/{filename}.png")
        plt.close()