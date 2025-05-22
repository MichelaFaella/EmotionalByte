import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def log_confusion_matrix(writer, labels, preds, epoch, phase):
    class_names = ['hap', 'sad', 'ang', 'neu', 'fru', 'sur']

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=class_names, cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"{phase} Confusion Matrix (Epoch {epoch})")
    writer.add_figure(f'{phase}/confusion_matrix', fig, epoch)
    plt.close(fig) 