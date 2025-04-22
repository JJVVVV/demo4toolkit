# demo4toolkit

# 修改 Trainer
文件位置: `path/to/miniconda3/envs/your_env/lib/python3.11/site-packages/transformers/trainer.py`
因为 toolkit.nlp 中的 TextDataset 需要为不同的 split 配置了不同的 collate_fn, 而 hf 的 trainer 只接受一个 data_collator，其训练和测试公用同一个 collate_fn。
为了使其能被 hf 的 trainer 使用，需要修改部分 trainer 的源码：
1. 在`Trainer.get_train_dataloader`函数中, 在代码
    ```python
    data_collator = self.data_collator
    ```
    后添加
    ```python
    if hasattr(self.train_dataset, "collate_fn"):
        data_collator = self.train_dataset.collate_fn
    ```
2. 在`Trainer.get_eval_dataloader`函数中, 在代码
    ```python
    data_collator = self.data_collator
    ```
    后添加
    ```python
    if hasattr(self.eval_dataset, "collate_fn"):
        data_collator = self.eval_dataset.collate_fn
    ```

# TODO
1. 当前 HF 没有支持 encoder-only 模型在训练中进行评估并计算 metric 的 Trainer. 暂时使用 Seq2SeqTrainer, 只能在完成训练后手动进行模型评估.