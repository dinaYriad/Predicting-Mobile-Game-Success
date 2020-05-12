import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

class Graph:
    def __init__(self, X=None, y=None, data=None, label=None):
        self.X = X
        self.y = y
        self.data = data
        self.label = label

    def feature_VS_feature_points(self):
        combinations = self.get_features_combinations(len(self.X.columns))

        for i in range(len(combinations)):
            feature1_i = combinations[i][0]
            feature2_i = combinations[i][1]

            feature1_name = self.X.columns[feature1_i]
            feature2_name = self.X.columns[feature2_i]

            sns.lmplot(x=feature1_name, y=feature2_name, data=self.data, hue=self.label,
                       legend=True, palette='Set1', fit_reg=False, scatter_kws={"s": 70})
            i += 1
            plt.show()

    def feature_VS_label_points(self):
        y = np.expand_dims(self.y, axis=1)
        for feature in self.X.columns:
            x = np.expand_dims(self.X[feature], axis=1)

            plt.title('Relating between ' + self.label + ' and ' + feature)
            plt.xlabel(feature, fontsize=20)
            plt.ylabel(self.label, fontsize=20)
            plt.scatter(x, y)

            plt.show()

    def get_features_combinations(self, nColumns):
        combinations = []
        for i in range(nColumns):
            for j in range(i + 1, nColumns):
                singleCombination = (i, j)
                combinations.append(singleCombination)

        return combinations

    def plot(self, title, xLabel, yLabel):
        plt.figure(figsize=(12, 6))
        plt.plot(self.X, self.y, color='green')
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

    def plot_pca(self):

        sns.lmplot(x='PCA 1', y='PCA 2', data=self.data, hue=self.label,
                   legend=True, palette='Set1', fit_reg=False, scatter_kws={"s": 70})
        plt.show()

class HeatMap:
    @staticmethod
    def show(content):
        sns.heatmap(content, annot=True)
        plt.show()
