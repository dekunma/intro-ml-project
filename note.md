- 12/06: resnet152 epoch 160-180 performance not better than that of 140. (maybe because of the learning rate decay or overfitting). Need to submit resnet_1_epoch_140 for test tomorrow.
- 12/07: need to try depth image global normalization vs. per-image normalization.
- 12/08: after having normalization, the performance of resnest269 was way better.
- 12/08: need to submit hrnet_1 epoch 200 and 210.
- 12/08: resnest269 using smoothl1loss, performance not able to increase after ~90 epochs, and performance was not as good as MSE loss. Need to submit resnet269's 185&175
- 12/12: submitted resnest269 [1, 7, 8] w/ the same config for tuning. Submitted convnext_4 with 100% training data. Submitted convnext_5 w/ fewer warmup epochs & 70% training data.
- 12/12: resnest269 [7, 8] did not get ideal result, resubmitted w/o loading pretrained weights.
- 12/13: RGB channel-wise norm, submitted resnest269 [1,7,8] and convnext [4]
- 12/13: Submitted resnest269 [1], checkpoint every epoch after 170. [11, 12] for more lr decay.