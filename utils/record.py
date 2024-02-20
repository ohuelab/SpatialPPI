import os
import matplotlib.pyplot as plt


def saveConfig(path, args):
    with open(os.path.join(path, "configs.txt"), 'w', encoding="utf-8") as f:
        f.write(f"model: {args.model}\n")
        f.write(f"datapath: {args.datapath}\n")
        f.write(f"weights: {args.weights}\n")
        f.write(f"savingPath: {args.savingPath}\n")
        f.write(f"train_set: {args.train_set}\n")
        f.write(f"test_set: {args.test_set}\n")
        f.write(f"augment: {args.augment}\n")
        f.write(f"batch: {args.batch}\n")
        f.write(f"alength: {args.alength}\n")
        f.write(f"ndims: {args.ndims}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epoch: {args.epoch}\n")
        f.write(f"lr: {args.lr}\n")


def drawFig(history, examine, monitor, savingPath):
    acc = history.history[examine]
    val_acc = history.history[monitor]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    with open(os.path.join(savingPath, "result.txt"), 'w') as f:
        for i in range(len(acc)):
            f.write(f"{i}\tacc:{round(acc[i], 2)}\tval_acc:{round(val_acc[i], 2)}\n")

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 3.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(savingPath, "result.jpg"))
