import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_checkpoint(self, trainer, pl_module):
        # 获取模型的指定参数
        print('opopplll')
        state_dict = pl_module.cond_stage_model.state_dict()
        
        # 创建模型检查点字典
        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'pytorch-lightning_version': pl.__version__,
            'state_dict': state_dict,
            'optimizer_states': {k: v.state_dict() for k, v in trainer.optimizers.items()},
            'lr_schedulers': {k: v.state_dict() for k, v in trainer.lr_schedulers.items()},
        }

        # 保存检查点
        self._save_checkpoint(trainer, checkpoint)