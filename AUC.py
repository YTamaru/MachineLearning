from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import csv
import pandas as pd

class AUC():
    def __init__(self, path):
        self.path = path

    def import_csv_AUC(self):
        df = pd.read_csv("self.path")
        true_labels = df[[0]]
        scores = df[[1]]
        return true_labels, scores

    def AUC(self):


