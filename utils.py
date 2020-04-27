import matplotlib.pyplot as plt
import torch.nn.functional as F


class ConfidenceHistograms:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        self.data = {
            "mcp_successes": [],
            "mcp_errors": [],
            "tcp_successes": [],
            "tcp_errors": [],
        }

    def step(self, label, proba, confidence):
        proba = F.softmax(proba, dim=1)
        pred = proba.argmax(1)
        self.data["mcp_successes"].extend((proba[pred == label].max(1)[0]).cpu().tolist())
        idx = (pred != label)
        if idx.sum() > 0:
            self.data["mcp_errors"].extend((proba[idx].max(1)[0]).cpu().tolist())
        self.data["tcp_successes"].extend(confidence[pred == label].view(-1).cpu().tolist())
        self.data["tcp_errors"].extend(confidence[pred != label].view(-1).cpu().tolist())

    def _create_histogram(self, mode="tcp"):
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(
            self.data[f"{mode}_successes"], color='g',
            bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
            label="successes")
        plt.hist(
            self.data[f"{mode}_errors"], color='r',
            bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
            alpha=.5, label="errors")
        plt.xlabel("confidence")
        plt.title("True class probability" if mode == "tcp" else "Maximum class probability")
        plt.legend()
        return fig

    def get_histograms(self):
        mcp_histogram = self._create_histogram(mode="mcp")
        tcp_histogram = self._create_histogram(mode="tcp")
        self._initialize()
        return mcp_histogram, tcp_histogram
