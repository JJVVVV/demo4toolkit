import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel, RobertaConfig, RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# from .tricks import generate_distribution, generate_distribution2, shift_embeddings, shift_embeddings_ball


class RobertaModel_multi_classify(RobertaModel):
    def __init__(self, config, num_classification: int, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classification)
        self.loss_func = nn.CrossEntropyLoss()
        self.vocab_size = self.config.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, num_classification, *model_args, **kwargs):
        kwargs["num_classification"] = num_classification
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = False,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.view(-1))
        ret["loss"] = loss

        return ret


class DoubleRoberta(PreTrainedModel):
    # config_class = RobertaConfig  # Specify the configuration class if needed
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize weights and biases as needed
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, config: PretrainedConfig, num_classification: int | None = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if num_classification is None:
            assert (
                hasattr(config, "num_classification") and config.num_classification is not None
            ), "`num_classification` is neither set in `config` nor set in `args` "
            num_classification = config.num_classification
        self.roberta1 = RobertaModel(config)
        self.roberta2 = RobertaModel(config)

        # Example: Adding multiple linear layers
        self.linear_layers = nn.ModuleDict(
            {
                # "dropout": nn.Dropout(config.hidden_dropout_prob),
                "pooler": nn.Linear(config.hidden_size * 2, config.hidden_size),
                "tanh": nn.Tanh(),
                "classifier": nn.Linear(config.hidden_size, num_classification),
            }
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = False,
    ):
        # Get outputs from both BERT models
        ret = dict()
        outputs = []
        for i in range(input_ids.shape[1]):
            outputs.append(
                getattr(self, f"roberta{i+1}")(
                    input_ids[:, i],
                    attention_mask[:, i],
                    token_type_ids[:, i] if token_type_ids is not None else None,
                    position_ids[:, i] if position_ids is not None else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            )
        cls1 = outputs[0].last_hidden_state[:, 0]
        cls2 = outputs[1].last_hidden_state[:, 0]
        # Concatenate or process outputs as needed
        combined_output = torch.cat((cls1, cls2), dim=1)
        # Pass through linear layer
        x = combined_output
        for name, layer in self.linear_layers.items():
            x = layer(x)
        ret["logits"] = x

        if labels is not None:
            loss = self.loss_func(x, labels.view(-1))
            ret["loss"] = loss
        # if labels is not None:
        #     klloss = kl_loss(cls1, cls2, temperature=1)
        #     bceloss = self.loss_func(x, labels.float())
        #     ret["loss"] = klloss + bceloss
        #     ret["loss_log"] = Tensor([bceloss, klloss])
        # import pdb
        # # pdb.set_trace()

        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        num_classification: int | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> PreTrainedModel:
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
        if num_classification is not None:
            config.num_classification = num_classification
        model = cls(config, num_classification=num_classification)

        # Load the BERT models
        if (pretrained_model_name_or_path / "roberta1").exists() and (pretrained_model_name_or_path / "roberta2").exists():
            model.roberta1 = RobertaModel.from_pretrained(pretrained_model_name_or_path / "roberta1")
            model.roberta2 = RobertaModel.from_pretrained(pretrained_model_name_or_path / "roberta2")
        else:
            model.roberta1 = RobertaModel.from_pretrained(pretrained_model_name_or_path)
            model.roberta2 = RobertaModel.from_pretrained(pretrained_model_name_or_path)

        # Attempt to load the model's state dict
        try:
            model.linear_layers.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/custom_model.bin"), strict=True)
        except FileNotFoundError:
            print("Warning: custom_model.bin not found. Initializing linear layers randomly.")
            # for layer in model.linear_layers.values():
            #     layer.apply(model._init_weights)  # Randomly initialize weights

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # Save the configuration and the two BERT models
        save_directory = Path(save_directory)
        self.config.save_pretrained(save_directory)
        self.roberta1.save_pretrained(save_directory / "roberta1", is_main_process=is_main_process, max_shard_size=max_shard_size)
        self.roberta2.save_pretrained(save_directory / "roberta2", is_main_process=is_main_process, max_shard_size=max_shard_size)
        if is_main_process:
            torch.save(self.linear_layers.state_dict(), save_directory / "custom_model.bin")


# ##############################################################################################################################################


class BertModel_multi_classify(BertModel):
    def __init__(self, config, num_classification, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classification)
        self.loss_func = nn.CrossEntropyLoss()
        self.vocab_size = self.config.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, num_classification, *model_args, **kwargs):
        kwargs["num_classification"] = num_classification
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
        # model.min_threshold = min_threshold

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = False,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.view(-1))
        ret["loss"] = loss

        return ret


class DoubleBERT(PreTrainedModel):
    # config_class = RobertaConfig  # Specify the configuration class if needed
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize weights and biases as needed
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, config: PretrainedConfig, num_classification: int | None = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if num_classification is None:
            assert (
                hasattr(config, "num_classification") and config.num_classification is not None
            ), "`num_classification` is neither set in `config` nor set in `args` "
            num_classification = config.num_classification
        self.bert_1 = BertModel(config)
        self.bert_2 = BertModel(config)

        # Example: Adding multiple linear layers
        self.linear_layers = nn.ModuleDict(
            {
                # "dropout": nn.Dropout(config.hidden_dropout_prob),
                "pooler": nn.Linear(config.hidden_size * 2, config.hidden_size),
                "tanh": nn.Tanh(),
                "classifier": nn.Linear(config.hidden_size, num_classification),
            }
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = False,
    ):
        # Get outputs from both BERT models
        ret = dict()
        outputs = []
        for i in range(input_ids.shape[1]):
            outputs.append(
                getattr(self, f"bert_{i+1}")(
                    input_ids[:, i],
                    attention_mask[:, i],
                    token_type_ids[:, i] if token_type_ids is not None else None,
                    position_ids[:, i] if position_ids is not None else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            )
        cls1 = outputs[0].last_hidden_state[:, 0]
        cls2 = outputs[1].last_hidden_state[:, 0]
        # Concatenate or process outputs as needed
        combined_output = torch.cat((cls1, cls2), dim=1)
        # Pass through linear layer
        x = combined_output
        for name, layer in self.linear_layers.items():
            x = layer(x)
        ret["logits"] = x

        if labels is not None:
            loss = self.loss_func(x, labels.view(-1))
            ret["loss"] = loss
        # if labels is not None:
        #     klloss = kl_loss(cls1, cls2, temperature=1)
        #     bceloss = self.loss_func(x, labels.float())
        #     ret["loss"] = klloss + bceloss
        #     ret["loss_log"] = Tensor([bceloss, klloss])
        # import pdb
        # # pdb.set_trace()

        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        num_classification: int | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> PreTrainedModel:
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        if num_classification is not None:
            config.num_classification = num_classification
        model = cls(config, num_classification=num_classification)

        # Load the BERT models
        if (pretrained_model_name_or_path / "roberta1").exists() and (pretrained_model_name_or_path / "roberta2").exists():
            model.bert_1 = BertModel.from_pretrained(pretrained_model_name_or_path / "roberta1")
            model.bert_2 = BertModel.from_pretrained(pretrained_model_name_or_path / "roberta2")
        else:
            model.bert_1 = BertModel.from_pretrained(pretrained_model_name_or_path)
            model.bert_2 = BertModel.from_pretrained(pretrained_model_name_or_path)

        # Attempt to load the model's state dict
        try:
            model.linear_layers.load_state_dict(torch.load(f"{pretrained_model_name_or_path}/custom_model.bin"), strict=True)
        except FileNotFoundError:
            print("Warning: custom_model.bin not found. Initializing linear layers randomly.")
            # for layer in model.linear_layers.values():
            #     layer.apply(model._init_weights)  # Randomly initialize weights

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # Save the configuration and the two BERT models
        save_directory = Path(save_directory)
        self.config.save_pretrained(save_directory)
        self.bert_1.save_pretrained(save_directory / "bert1", is_main_process=is_main_process, max_shard_size=max_shard_size)
        self.bert_2.save_pretrained(save_directory / "bert2", is_main_process=is_main_process, max_shard_size=max_shard_size)
        if is_main_process:
            torch.save(self.linear_layers.state_dict(), save_directory / "custom_model.bin")


# ###################################################################################################################################################
