import json
import re
import time
from pathlib import Path

import toolkit
import torch
import torch.distributed as dist
from fire import Fire
from toolkit.enums import Split
from toolkit.logger import getLogger
from toolkit.nlp import NLPTrainingConfig, TextDataset
from toolkit.training.initializer import initialize
from toolkit.training.trainer import Trainer
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from models.classification_models import BertModel_multi_classify, DoubleBERT, DoubleRoberta, RobertaModel_multi_classify
from models.generation_models import LlamaForCausalLM_baseline
from utils.evaluators import Evaluator4Classify, Evaluator4Generate
from utils.load_data_fn import load_data_fn4classify, load_data_fn4generate

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_tokenizer() -> PreTrainedTokenizer:
    # * Load tokenizer
    tokenizer_kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, **tokenizer_kwargs, trust_remote_code=True)
    if dist.is_initialized():
        dist.barrier()
    return tokenizer


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
    return train_dataset, val_dataset, test_dataset


def load_hf_generation_config(model, config):
    from pprint import pp

    if hasattr(config, "hf_generation_config_file"):
        with open(config.hf_generation_config_file, "r") as f:
            data = f.read()
            d = json.loads(data)
            # print(d)
            # hf_generate_config = GenerationConfig.from_dict(d)
            if model is not None:
                from copy import deepcopy

                hf_generate_config = deepcopy(model.generation_config)
            else:
                hf_generate_config = GenerationConfig.from_pretrained(config.model_dir)
            hf_generate_config.update(**d)
            print("\nhf generation config:")
            pp(hf_generate_config)
            return hf_generate_config


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

    # * load generation config
    generation_config = load_hf_generation_config(model, config)

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # * Train
    trainer = Trainer(
        config=config,
        model=model,
        model_config=model_config,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        extral_evaluators=[Evaluator4Generate] if config.task_type == "generate" else [Evaluator4Classify],
        optimizer=config.opt_type,
        scheduler=config.sch_type,
        tokenizer=tokenizer,
        dashboard_writer=run,
        callback_load_model=load_model,
        # extral_args_training={"is_train": True},
        extral_args_evaluation={"generation_config": generation_config} if config.task_type == "generate" else None,
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
    # toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    local_rank, world_size = Trainer.initialize(config, 0.5 if config.parallel_mode is None else None)
    print("local_rank: ", local_rank, "world_size: ", world_size)

    # * Create tensorboard writer
    if config.dashboard is None:
        run = None
        main()
    else:
        if local_rank == 0:
            if config.dashboard == "wandb":
                import wandb

                with wandb.init(
                    # mode="disabled",
                    project="second",
                    config=config.to_dict(),
                    group=f"{config.dataset_name},train_data={config.part}",
                    tags=[config.dataset_name, config.model_type, config.model_name, config.text_type],
                ) as run:
                    assert run is wandb.run
                    main()
            elif config.dashboard == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                run_dir = Path("tensorboard", _dir, "logs")
                run_dir.mkdir(parents=True, exist_ok=True)
                with SummaryWriter(comment="training", log_dir=run_dir) as run:
                    main()
        else:
            main()
