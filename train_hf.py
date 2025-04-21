import json
import re
import time
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import toolkit
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from fire import Fire
from toolkit.logger import getLogger
from toolkit.nlp import NLPTrainingConfig, TextDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from models.classification_models import BertModel_multi_classify, DoubleBERT, DoubleRoberta, RobertaModel_multi_classify
from utils.evaluators import Evaluator4Classify, Evaluator4Generate
from utils.load_data_fn import load_data_fn4classify, load_data_fn4generate

# import evaluate
# metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# def load_dataset(tokenizer: PreTrainedTokenizer) -> tuple:
#     # * Load training data, development data and test data
#     df_train = pd.read_json(config.train_file_path, lines=False)
#     df_dev = pd.read_json(config.val_file_path, lines=False)

#     # 将 Pandas DataFrame 转换为 Hugging Face Dataset
#     train_dataset = Dataset.from_pandas(df_train)
#     dev_dataset = Dataset.from_pandas(df_dev)

#     # 可选：将训练集和验证集组合成一个 DatasetDict
#     dataset_dict = DatasetDict({"train": train_dataset, "val": dev_dataset})

#     # 打印数据集信息
#     print(dataset_dict)
#     if dist.is_initialized():
#         dist.barrier()
#     return dataset_dict


def load_dataset(tokenizer: PreTrainedTokenizer) -> tuple:
    # * Load training data, development data and test data
    train_dataset = TextDataset.from_file(
        tokenizer=tokenizer,
        load_data_fn=load_data_fn4generate if config.task_type == "generate" else load_data_fn4classify,
        split="TRAINING",
        configs=config,
        config_load_data=config,
    )
    val_dataset = TextDataset.from_file(
        tokenizer=tokenizer,
        load_data_fn=load_data_fn4generate if config.task_type == "generate" else load_data_fn4classify,
        split="VALIDATION",
        configs=config,
        config_load_data=config,
    )
    test_dataset = TextDataset.from_file(
        tokenizer=tokenizer,
        load_data_fn=load_data_fn4generate if config.task_type == "generate" else load_data_fn4classify,
        split="TEST",
        configs=config,
        config_load_data=config,
    )
    if dist.is_initialized():
        dist.barrier()
    return DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})


def load_tokenizer() -> PreTrainedTokenizer:
    # * Load tokenizer
    tokenizer_kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, **tokenizer_kwargs, trust_remote_code=True)
    if dist.is_initialized():
        dist.barrier()
    return tokenizer


def load_model(tokenizer):
    start = time.time()

    # * define model class
    if config.task_type == "generate":
        ModelClass = AutoModelForCausalLM
    else:
        ModelClass = BertModel_multi_classify
    logger.info(f"local_rank {local_rank}: {str(ModelClass)}")

    # * Determine the model dir
    pretrained_model_dir = config.model_dir if config.model_dir is not None else config.model_type
    logger.debug(f"local_rank {local_rank}: load model from {pretrained_model_dir}")

    # * Load model
    if ModelClass in (BertModel_multi_classify, RobertaModel_multi_classify, DoubleRoberta, DoubleBERT):
        model = ModelClass.from_pretrained(pretrained_model_dir, torch_dtype=config.torch_dtype, num_classification=config.num_classification)
    else:
        model = ModelClass.from_pretrained(pretrained_model_dir, torch_dtype=config.torch_dtype)

    # * add pad token
    # print(tokenizer.pad_token, tokenizer.eos_token)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.debug(f"Setting pad token to eos token: {tokenizer.eos_token}, {tokenizer.eos_token_id}")
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug(f"The pad token and pad token id: {tokenizer.pad_token}, {tokenizer.pad_token_id}")
        else:
            logger.debug(f"len(tokenizer): {len(tokenizer)}")
            logger.debug(f"Adding pad token {DEFAULT_PAD_TOKEN} ...")
            tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
            logger.debug(f"len(tokenizer): {len(tokenizer)}")
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) != embedding_size:
                logger.debug(f"len(tokenizer)!=embedding_size: {len(tokenizer)}!={embedding_size}")
                logger.debug("resize the embedding size by the size of the tokenizer ...")
                model.resize_token_embeddings(len(tokenizer))

    # * PEFT
    if "lora" in config.model_name:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Conv1D

        def get_specific_layer_names(model):
            # Create a list to store the layer names
            layer_names = []

            # Recursively visit all modules and submodules
            for name, module in model.named_modules():
                # Check if the module is an instance of the specified layers
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                    # model name parsing
                    if x := ".".join(name.split(".")[4:]).split(".")[0]:
                        layer_names.append(x)

            return layer_names

        logger.info(str(list(set(get_specific_layer_names(model)))))

        peft_config = LoraConfig(
            # target_modules = 'all-linear',
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules="all",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # # ? 用ds时该方法可以直接加载模型到GPU上，但是如何无法适配PEFT和resize_embedding
    # if config.parallel_mode != "deepspeed":
    #     if ModelClass in (BertModel_multi_classify, RobertaModel_multi_classify, DoubleRoberta, DoubleBERT):
    #         model = ModelClass.from_pretrained(pretrained_model_dir, torch_dtype=config.torch_dtype, num_classification=config.num_classification)
    #     else:
    #         model = ModelClass.from_pretrained(pretrained_model_dir, torch_dtype=config.torch_dtype)
    # else:
    #     logger.debug(f"local_rank {local_rank}: Construct `from_pretrained` kwargs ...")
    #     from_pretrained_kwargs = dict(torch_dtype=config.torch_dtype, trust_remote_code=True)
    #     model = None
    #     model_config = AutoConfig.from_pretrained(config.model_dir)
    #     model = AutoModelForCausalLM.from_config(model_config)

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.debug(f"Total model size: {n_params/2**20:.2f}M params")

    end = time.time()
    logger.debug(f"local_rank {local_rank}: Loading model takes {end - start:.2f} sec.")

    # * sync
    if dist.is_initialized():
        dist.barrier()
    return model


def compute_metrics(p: EvalPrediction):
    # evaluator_class = Evaluator4Generate if config.task_type == "generate" else Evaluator4Classify
    # evaluator = evaluator_class(task_type=config.task_type, split="VALIDATION", config=config, model=True, dataset=True)
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    match config.task_type:
        case "regress":
            metric = evaluate.load("mse", cache_dir=config.save_dir)
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        case "classify":
            metric = evaluate.load("accuracy", cache_dir=config.save_dir)
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        case "generate":
            pass
        case "multi_label":  # ? NLPTrainingConfig还未支持这种任务
            metric = evaluate.load("f1", config_name="multilabel", cache_dir=config.save_dir)
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


def main() -> None:
    # * Loading tokenizer
    tokenizer = load_tokenizer()

    # *load model
    if config.parallel_mode != "deepspeed":
        model = load_model(tokenizer)
        model_config = None
    else:
        model = None
        model_config = ...

    # # * load generation config
    # generation_config = load_hf_generation_config(model, config)

    # * load dataset
    dataset = load_dataset(tokenizer)

    # * Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        processing_class=tokenizer,
        data_collator=dataset["train"].collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    #  todo 逻辑有问题，只能手动加载ckpt中的模型，因为cpkt_manager在trainer中，而trainer的初始化在模型加载之后。
    config.model_dir = config.model_dir if trainer.ckpt_manager.latest_id < 0 else trainer.ckpt_manager.latest_dir


if __name__ == "__main__":
    # * Get args
    config: NLPTrainingConfig = Fire(NLPTrainingConfig, silence=True)

    # * search model
    if config.model_dir is None:
        candidate_dirs = [Path("/data/jjwang/pretrained/"), Path("/public/home/hongy/pretrained_models/")]
        for d in candidate_dirs:
            if (d / config.model_type).exists():
                config.model_dir = d / config.model_type
                break
    assert config.model_dir is not None

    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        config.dataset_name,
        config.model_type,
        config.text_type,
        # f"{config.data_file_name_train.split('.')[0]}-{config.data_file_name_val.split('.')[0]}-{str(config.data_file_name_test).split('.')[0]}",
        f"{config.train_file_path.stem}-{config.val_file_path.stem if config.val_file_path else None}-{config.test_file_path.stem if config.test_file_path else None}",
        str(config.part),
        config.model_name,
        str(config.epochs),
        str(config.train_batch_size),
        str(config.opt_lr),
        str(config.seed),
    )
    if config.save_dir is None:
        config.save_dir = Path("outputs", _dir)
    config.save(config.save_dir, silence=False)

    # * Create logger
    output_path_logger = config.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    toolkit.set_file_logger(output_path_logger)

    # parser = HfArgumentParser(TrainingArguments)
    # training_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(
        output_dir=config.save_dir,
        learning_rate=config.opt_lr,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.infer_batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.opt_weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=False,
        logging_steps=config.logging_steps,
    )
    if dist.is_initialized():
        local_rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, world_size = 0, 1
    main()
