from module import LaneClassification
import pytorch_lightning as pl

model = LaneClassification.load_from_checkpoint(checkpoint_path='./lightning_logs/version_5/checkpoints/epoch=4-step=102610.ckpt',
                                                map_location='cuda:0')
model.batch_size = 4
trainer = pl.Trainer(accelerator='gpu', devices=1)

print(trainer.test(model))