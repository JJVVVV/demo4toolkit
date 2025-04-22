# demo4toolkit

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
2. 在`Trainer.get_test_dataloader`函数中, 在代码
    ```python
    data_collator = self.data_collator
    ```
    后添加
    ```python
    if hasattr(self.eval_dataset, "collate_fn"):
        data_collator = self.eval_dataset.collate_fn
    ```