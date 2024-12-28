import copy
import fcntl
import json
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score
from toolkit import toolkit_logger
from toolkit.enums import Split
from toolkit.metric import MetricDict, bleu, rouge, self_bleu
from toolkit.training.evaluator import Evaluator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Evaluator4Generate(Evaluator):
    write_file = True

    def calculate_metric_callback(self, all_labels: list[str], all_logits: list[str], mean_loss: float) -> MetricDict:
        if self.write_file:
            self.save_eval_result(all_labels, all_logits)
        metric = (rouge(all_logits, all_labels, "zh", ("rougeL")) * 100).round(2)
        return metric


class Evaluator4Classify(Evaluator):
    write_file = True

    def calculate_metric_callback(self, all_labels: list[str], all_logits: list[str], mean_loss: float) -> MetricDict:
        if self.write_file:
            self.save_eval_result(all_labels, all_logits)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        # all_preds = (all_logits > 0).astype(int)
        all_preds = np.argmax(all_logits, axis=1, keepdims=True)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")
        metric = MetricDict({"accuracy": acc * 100, "F1-score": f1 * 100, "loss": mean_loss}).round(2)
        return metric
