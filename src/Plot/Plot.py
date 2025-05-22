import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os

Plot_save_result = "../Result/Plots/"

def plotLosses(logs, epochs, save_path="", hyperparams=None):
    epoch_range = list(range(1, epochs + 1))

    # --- Task and Cross-Entropy Losses ---
    fig_ce, ax_ce = plt.subplots(figsize=(10, 4))
    ax_ce.plot(epoch_range, [l.item() for l in logs['loss_task']], label='loss_task')
    ax_ce.plot(epoch_range, [l.item() for l in logs['loss_ce_t']], label='loss_ce_text')
    ax_ce.plot(epoch_range, [l.item() for l in logs['loss_ce_a']], label='loss_ce_audio')
    ax_ce.set_xlabel("Epochs")
    ax_ce.set_ylabel("Loss")
    ax_ce.set_title("TRAIN - Task and Cross-Entropy Losses")
    ax_ce.legend()
    ax_ce.grid(True)

    if hyperparams:
        ax_ce.text(0.5, -0.35, f"Hyperparameters: {hyperparams}",
                   ha='center', va='top', transform=ax_ce.transAxes, fontsize=9)

    fig_ce.tight_layout()
    ShowOrSavePlot(Plot_save_result + save_path, "TRAIN_loss_ce")
    plt.close(fig_ce)

    # --- KL Divergence Losses ---
    fig_kl, ax_kl = plt.subplots(figsize=(10, 4))
    ax_kl.plot(epoch_range, [l.item() for l in logs['loss_kl_t']], label='loss_kl_text')
    ax_kl.plot(epoch_range, [l.item() for l in logs['loss_kl_a']], label='loss_kl_audio')
    ax_kl.set_xlabel("Epochs")
    ax_kl.set_ylabel("KL Loss")
    ax_kl.set_title("TRAIN - KL Divergence Losses")
    ax_kl.legend()
    ax_kl.grid(True)

    if hyperparams:
        ax_kl.text(0.5, -0.35, f"Hyperparameters: {hyperparams}",
                   ha='center', va='top', transform=ax_kl.transAxes, fontsize=9)

    fig_kl.tight_layout()
    ShowOrSavePlot(Plot_save_result + save_path, "TRAIN_loss_kl")
    plt.close(fig_kl)



def plotEval(logs, epochs, phase, save_path="", hyperparams=None):
    """
    phase: should be either "VALIDATION" or "TEST"
    """
    epoch_range = list(range(1, epochs + 1))


    # --- Fscore and Accuracy---
    fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
    ax_eval.plot(epoch_range, logs['fscore'], label=f'{phase.lower()}_fscore')
    ax_eval.plot(epoch_range, logs['acc'], label=f'{phase.lower()}_accuracy')
    ax_eval.set_xlabel("Epochs")
    #ax_eval.set_ylabel("Fscore")
    ax_eval.set_title(f"{phase} - Accuracy and Fscore")
    ax_eval.legend()
    ax_eval.grid(True)

    if hyperparams:
        ax_eval.text(0.5, -0.35, f"Hyperparameters: {hyperparams}",
                    ha='center', va='top', transform=ax_eval.transAxes, fontsize=9)

    fig_eval.tight_layout()
    ShowOrSavePlot(Plot_save_result + save_path, f"{phase}_eval")
    plt.close(fig_eval)


def plotTotalLoss(logs_train, logs_test, epochs, save_path="", hyperparams=None):
    epoch_range = list(range(1, epochs + 1))

    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(epoch_range, [l.item() for l in logs_train['loss']], label='Train_loss')
    ax_loss.plot(epoch_range, logs_test['loss'], label=f'Test_loss')
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Total Loss")
    ax_loss.set_title(f"Total Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    if hyperparams:
        ax_loss.text(0.5, -0.35, f"Hyperparameters: {hyperparams}",
                      ha='center', va='top', transform=ax_loss.transAxes, fontsize=9)

    fig_loss.tight_layout()
    ShowOrSavePlot(Plot_save_result + save_path, f"TOTAL_loss")
    plt.close(fig_loss)

def confusionMatrix(labels, preds, save_path=""):
    cfm = metrics.confusion_matrix(labels, preds)
    labels_name = ['hap\nexc', 'sad', 'ang', 'neu', 'fru', 'sur, fea\ndis, oth']
    disp = metrics.ConfusionMatrixDisplay(cfm, display_labels=labels_name)
    plt.figure(figsize=(15, 15))
    disp.plot(cmap='Blues')
    plt.tight_layout()
    ShowOrSavePlot(Plot_save_result + save_path, f"Confusion Matrix")


def ShowOrSavePlot(path: str = None, filename: str = "img"):
    """
    Saves the current plot to a file or displays it if no path is specified.

    :param path: Directory where the plot should be saved. If None or empty, the plot is displayed instead.
    :param filename: Name of the file (without extension). Defaults to "img".
    """
    if not path:
        plt.show()
        return

    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Construct the full file path
    filepath = os.path.join(path, f"{filename}.png")

    # Save and close the current figure
    plt.savefig(filepath)
    plt.close()
