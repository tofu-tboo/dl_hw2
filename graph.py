import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class LossGraph:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
    
    def show(self):
        plt.plot(self.train_losses, label="train")
        plt.plot(self.test_losses, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss graph")
        plt.legend()
        plt.show()

class ConfusionMatrix:
    def __init__(self):
        self.mat = np.zeros((5, 5))

    def show(self):
        plt.figure(figsize=(4,4))
        sns.heatmap(self.mat / self.mat.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues")
        plt.ylabel("Prediction")
        plt.show()