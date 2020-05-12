import matplotlib.pyplot as plt
import seaborn as sns

class Correlation:
    def __init__(self, data):
        self.data = data

    def correlate(self):
        df_corr = self.data.corr()
        ax = sns.heatmap(df_corr)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()