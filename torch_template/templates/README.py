# {{name}}
This repo is automatically generated by `torch-template`.

### Requirements

```yaml
python >= 3.5
torch >= 0.4
tensorboardX >= 1.6
utils-misc >= 0.0.3
torch-template >= 0.0.4
```

### Setup

```bash
pip3 install -r requirements.txt
```

### Train your own network
```bash
    python3 train.py --tag tag_1 --dataset voc --batch_size 16 --epochs 100 [--load <pretrained models folder> --which-epoch 500] --gpu_ids 0
```

### Test the model
```shell script
    python3 test.py --tag tag_1 --dataset voc --load <pretrained models folder> --which-epoch 500  # text results will be saved in 'results/tag_1' directory
```

### Visulization
```shell script
   tensorboard --logdir logs/tag_1
```
