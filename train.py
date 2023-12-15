
import pytorch_lightning as pl
import torch, random
from module import LaneClassification
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


torch.manual_seed(777)
random.seed(777)

model = LaneClassification()
model.batch_size = 16
trainer = pl.Trainer(accelerator='gpu', devices=2, max_epochs=5, strategy='DDP', check_val_every_n_epoch=1,
                     enable_progress_bar=True, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

trainer.fit(model)
