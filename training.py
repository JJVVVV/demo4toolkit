import re
import time
from pathlib import Path

import numpy as np
import torch
import json

# import wandb
from fire import Fire
from toolkit.enums import Split
from toolkit.logger import getLogger
from toolkit.nlp import NLPTrainingConfig, TextDataset
from toolkit.training.trainer import Trainer
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    GenerationConfig,
)


from models.classification_models import RobertaModel_multi_classify
from models.generation_models import LlamaForCausalLM_baseline
from utils.load_data_fns import LOAD_DATA_FNS, DatasetName, TextType
from utils.evaluators import Evaluator4Classify, Evaluator4Generate

# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_PAD_TOKEN = "<unk>"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"

# from toolkit.logger import _getLogger
# Evaluator1.logger = _getLogger("Eva")


def load_dataset(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> tuple:
    # * Load training data, development data and test data
    train_dataset = TextDataset.from_file(
        tokenizer=tokenizer,
        load_data_fn=LOAD_DATA_FNS[DATASETNAME],
        split=Split.TRAINING,
        configs=configs,
        text_type=TEXTTYPE,
        dataset_name=DATASETNAME,
        part=configs.part,
        model_name=configs.model_name,
    )
    # import pdb

    # from torch.utils.data import DataLoader

    # for batch in DataLoader(train_dataset, batch_size=4, collate_fn=train_dataset.collate_fn):
    #     print(batch)
    #     pdb.set_trace()
    #     break
    val_dataset = TextDataset.from_file(
        tokenizer=tokenizer,
        load_data_fn=LOAD_DATA_FNS[DATASETNAME],
        split=Split.VALIDATION,
        configs=configs,
        text_type=TEXTTYPE,
        dataset_name=DATASETNAME,
        part=configs.part,
        model_name=configs.model_name,
    )

    # for batch in DataLoader(val_dataset, batch_size=4, collate_fn=val_dataset.collate_fn):
    #     print(batch)
    #     pdb.set_trace()
    #     break
    try:
        test_dataset = TextDataset.from_file(
            tokenizer=tokenizer,
            load_data_fn=LOAD_DATA_FNS[DATASETNAME],
            split=Split.TEST,
            configs=configs,
            text_type=TEXTTYPE,
            dataset_name=DATASETNAME,
            part=configs.part,
            model_name=configs.model_name,
        )
    except TypeError as e:
        if local_rank == 0:
            logger.warning(e)
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


def load_hf_generation_config(model, config):
    if hasattr(config, "hf_gen_config_file"):
        with open(config.hf_gen_config_file, "r") as f:
            data = f.read()
            d = json.loads(data)
            print(d)
            # hf_generate_config = GenerationConfig.from_dict(d)
            if model is not None:
                from copy import deepcopy

                hf_generate_config = deepcopy(model.generation_config)
            else:
                hf_generate_config = GenerationConfig.from_pretrained(config.model_dir)
            hf_generate_config.update(**d)
            print("\nhf generation config:")
            print(hf_generate_config)
            return hf_generate_config


def load_model() -> tuple[PreTrainedModel | DDP, PreTrainedTokenizer | PreTrainedTokenizerFast, int]:
    # * define model class
    if configs.task_type == "generate":
        ModelClass = LlamaForCausalLM_baseline
    else:
        ModelClass = RobertaModel_multi_classify

    logger.info(f"local_rank {local_rank}: {str(ModelClass)}")

    # * Determine the model path
    pretrainedModelDir = configs.model_dir if configs.model_dir is not None else configs.model_type
    logger.debug(f"local_rank {local_rank}: load model from {pretrainedModelDir}")

    # * Load model, tokenizer to CPU memory
    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer to CPU memory...")
    start = time.time()
    # 加载自定义配置
    my_config = None
    try:
        my_config = AutoConfig.from_pretrained(f"config/my_{configs.model_type}_config")
        logger.debug(str(my_config))
    except:
        pass

    if ModelClass in (RobertaModel_multi_classify,):
        model = ModelClass.from_pretrained(pretrainedModelDir, config=my_config, num_classification=configs.num_classification)
    else:
        model = ModelClass.from_pretrained(pretrainedModelDir, config=my_config, torch_dtype=configs.torch_dtype)

    # tokenizer = AutoTokenizer.from_pretrained(pretrainedModelDir, do_lower_case=configs.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(pretrainedModelDir)
    end = time.time()

    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer from disk to CPU memory takes {end - start:.2f} sec.")

    # # * Load model to GPU memory
    # logger.debug(f"local_rank {local_rank}: Loading model to GPU memory...")
    # start = time.time()
    # model = model.cuda()
    # end = time.time()
    # logger.debug(f"local_rank {local_rank}: Loading model from CPU memory to GPU memory takes {end - start:.2f} sec.")

    # * add pad token
    if tokenizer.pad_token is None:
        # logger.debug(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        # tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
        logger.debug(f"Setting pad token to eos token: {tokenizer.eos_token}, {tokenizer.eos_token_id}")
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(f"The pad token and pad token id: {tokenizer.pad_token}, {tokenizer.pad_token_id}")
    logger.debug(f"len(tokenizer): {len(tokenizer)}")

    # # flan-t5 没有 ‘{’和‘}’
    # if isinstance(model, T5ForConditionalGeneration):
    #     tokenizer.add_tokens(["{", "}"])
    # logger.debug(f"len(tokenizer):{len(tokenizer)}")

    # # * resize embedding
    # if hasattr(model, "roberta1"):
    #     embedding_size = model.roberta1.get_input_embeddings().weight.shape[0]
    # else:
    #     embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) != embedding_size:
    #     logger.debug(f"len(tokenizer):{len(tokenizer)}")
    #     logger.debug(f"embedding_size: {embedding_size}")
    #     logger.debug("resize the embedding size by the size of the tokenizer")
    #     model.resize_token_embeddings(len(tokenizer))

    # * PEFT
    if "lora" in configs.model_name:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules="all",
            task_type=TaskType.SEQ_CLS,
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
        # print(model.score.weight.requires_grad)
        # model.score.weight.requires_grad = True

    return model, tokenizer


@record
def main() -> None:
    # * Loading model
    model, tokenizer = load_model()
    if configs.dashboard == "wandb":
        wandb.watch(model.module if hasattr(model, "module") else model, log_freq=256)

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # * Train
    trainer = Trainer(
        config=configs,
        model=model,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        extral_evaluators=[Evaluator4Classify] if configs.task_type == "classify" else [Evaluator4Generate],
        optimizer=configs.opt_type,
        scheduler=configs.sch_type,
        tokenizer=tokenizer,
        dashboard_writer=run,
        extral_args_training={"is_train": True},
        # extral_args_evaluation={"is_train": False},
    )
    trainer.train()
    # time.sleep(3)


if __name__ == "__main__":
    # * Get args
    configs: NLPTrainingConfig = Fire(NLPTrainingConfig, silence=True)
    # print(configs.shuffle)

    # * search model
    if configs.model_dir is None:
        candidate_dirs = [Path("/data/jjwang/pretrained/"), Path("/public/home/hongy/pretrained_models/")]
        for d in candidate_dirs:
            if (d / configs.model_type).exists():
                configs.model_dir = d / configs.model_type
                break
    assert configs.model_dir is not None

    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        configs.dataset_name,
        configs.model_type,
        configs.text_type,
        f"{configs.train_file_path.stem}-{configs.val_file_path.stem}-{configs.test_file_path.stem if configs.test_file_path else None}",
        str(configs.part),
        configs.model_name,
        str(configs.epochs),
        str(configs.train_batch_size),
        str(configs.opt_lr),
        str(configs.seed),
    )
    configs.save_dir = Path("outputs", _dir)
    configs.save(configs.save_dir, silence=False)

    # * Create logger
    output_path_logger = configs.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    # Evaluator1.logger = logger
    # toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    # local_rank, world_size = Trainer.initialize(configs)
    local_rank, world_size = Trainer.initialize(configs, 0.8 if configs.parallel_mode is None else None)
    print("local_rank: ", local_rank, "world_size: ", world_size)

    # * Global variable
    DATASETNAME = DatasetName[configs.dataset_name]
    TEXTTYPE = TextType[configs.text_type]

    # * Create tensorboard writer
    if configs.dashboard is None:
        run = None
        main()
    else:
        if local_rank == 0:
            if configs.dashboard == "wandb":
                with wandb.init(
                    # mode="disabled",
                    project="second",
                    config=configs.to_dict(),
                    group=f"{configs.dataset_name},train_data={configs.part}",
                    tags=[configs.dataset_name, configs.model_type, configs.model_name, configs.text_type],
                ) as run:
                    assert run is wandb.run
                    main()
            elif configs.dashboard == "tensorboard":
                run_dir = Path("tensorboard", _dir, "logs")
                run_dir.mkdir(parents=True, exist_ok=True)
                with SummaryWriter(comment="training", log_dir=run_dir) as run:
                    main()
        else:
            main()
