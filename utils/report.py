import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(prediction, target, args, ignored_labels, labels_text):
    """
        Compute Confusion Matrix, OA, PA, AA, Kappa
    """
    mask = np.ones(target.shape[:2], dtype=np.bool_)
    for k in ignored_labels:
        mask[target == k] = False
    target = target[mask]
    pred = prediction[mask]

    # compute Confusion Matrix
    cm = confusion_matrix(target, pred, labels=range(args.class_num))

    # compute Overall Accuracy (OA)
    oa = 1. * np.trace(cm) / np.sum(cm)

    # compute Producer Accuracy (PA)
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(args.class_num)])

    # compute Average Accuracy (AA)
    aa = np.mean(pa)

    # compute kappa coefficient
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)

    # Console output
    text = ""
    text += "Confusion matrix:\n"
    text += str(cm)
    text += "\n---\n"

    text += "Overall Accuracy: {:.04f}\n".format(oa)
    text += "---\n"

    text += "Producer's Accuracy:\n"
    for label, acc in zip(labels_text, pa):
        text += "\t{}: {:.04f}\n".format(label, acc)
    text += "---\n"

    text += "Average Accuracy: {:.04f}\n".format(aa)
    text += "---\n"

    text += "Kappa: {:.04f}\n".format(kappa)
    text += "---\n"

    if oa > args.best_acc:
        print(text)

    # 在wandb输出混淆矩阵的可视化图
    # cm = pd.DataFrame(data=cm / np.sum(cm, axis=1, keepdims=True), index=labels_text, columns=labels_text)
    # plt.figure(figsize=(12, 7))
    # Img = wandb.Image(sns.heatmap(data=cm, annot=True).get_figure(), caption=f"Confusion Matrix {0}")
    # wandb.log({"Confusion Matrix": Img})

    return oa, aa, kappa
