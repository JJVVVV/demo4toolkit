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
from toolkit.training import Evaluator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Evaluator4Generate(Evaluator):
    write_file = True

    def calculate_metric_callback(self, all_labels: list[str], all_logits: list[str], mean_loss: float, all_prompts=None) -> MetricDict:
        if self.write_file:
            self.save_eval_result(all_labels, all_logits)
        metric = (rouge(all_logits, all_labels, "zh", ("rougeL")) * 100).round(2)
        metric.round(2)
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
        if all_logits.shape[-1] == 2:
            f1 = f1_score(all_labels, all_preds, average="binary")
        else:
            f1 = f1_score(all_labels, all_preds, average="micro")
        metric = MetricDict({"accuracy": acc * 100, "F1-score": f1 * 100, "loss": mean_loss}).round(2)
        return metric


# class Evaluator4Generate(Evaluator):
#     write_file = True
#     input_output_label_path = "input_output_label.json"

#     def calculate_metric_callback(self, all_labels: list[str], all_logits: list[str], mean_loss: float) -> MetricDict:
#         local_rank = dist.get_rank() if dist.is_initialized() else 0
#         world_size = dist.get_world_size() if dist.is_initialized() else 1

#         if local_rank == 0:
#             objects4sync = []
#             dict_list = []
#             for idx, (a_label, a_logits) in enumerate(zip(all_labels, all_logits)):
#                 a = {"idx": idx, "label": a_label, "pred": a_logits}
#                 dict_list.append(a)
#             dict_list = sorted(dict_list, key=lambda x: x["idx"])
#             if self.write_file:
#                 generate_result_path: Path = (
#                     self.config.save_dir
#                     / "evaluators"
#                     / f"epoch={self.config.training_runtime['cur_epoch']:03d}_step={self.config.training_runtime['cur_step']}"
#                     / self.split.name
#                     / self.input_output_label_path
#                 )
#                 generate_result_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(generate_result_path, "w") as f:
#                     json.dump(dict_list, f, indent=2, ensure_ascii=False)
#             metric = rouge([it["pred"] for it in dict_list], [it["label"] for it in dict_list], "en", ("rougeL")) * 100
#             metric.round(2)
#             objects4sync.append(metric)
#         else:
#             objects4sync = [None]
#         if world_size > 1:
#             dist.broadcast_object_list(objects4sync, src=0)
#         return objects4sync[0]


# class Evaluator4Classify(Evaluator):
#     write_file = True
#     input_output_label_path = "input_output_label.json"

#     def calculate_metric_callback(self, all_labels: list[str], all_logits: list[str], mean_loss: float) -> MetricDict:
#         local_rank = dist.get_rank() if dist.is_initialized() else 0
#         world_size = dist.get_world_size() if dist.is_initialized() else 1

#         if local_rank == 0:
#             objects4sync = []
#             dict_list = []
#             for idx, (a_label, a_logits) in enumerate(zip(all_labels, all_logits)):
#                 a = {"idx": idx, "label": a_label, "pred": a_logits}
#                 dict_list.append(a)
#             dict_list = sorted(dict_list, key=lambda x: x["idx"])
#             if self.write_file:
#                 generate_result_path: Path = (
#                     self.config.save_dir
#                     / "evaluators"
#                     / f"epoch={self.config.training_runtime['cur_epoch']:03d}_step={self.config.training_runtime['cur_step']}"
#                     / self.split.name
#                     / self.input_output_label_path
#                 )
#                 generate_result_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(generate_result_path, "w") as f:
#                     json.dump(dict_list, f, indent=2, ensure_ascii=False)
#             all_labels = np.array(all_labels)
#             all_logits = np.array(all_logits)
#             all_preds = (all_logits > 0).astype(int)
#             acc = accuracy_score(all_labels, all_preds)
#             f1 = f1_score(all_labels, all_preds, average="binary")
#             metric = MetricDict({"accuracy": acc * 100, "F1-score": f1 * 100, "loss": mean_loss})
#             metric.round(2)
#             objects4sync.append(metric)
#         else:
#             objects4sync = [None]
#         if world_size > 1:
#             dist.broadcast_object_list(objects4sync, src=0)
#         return objects4sync[0]
