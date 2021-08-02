# train-CLIP-FT 📎

A PyTorch Lightning solution to finetuning the released CLIP models

## Dependencies
```bash
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install pytorch-lightning
```
 
## Usage 🚂

### Finetuning 🚆
To finetune an existing model all you have to do is change the string which specifies which model to load!

```python
clp, preprocess = clip.load("ViT-B/16", device='cpu')
```

Then simply type the following into your terminal:
```
python train_finetune.py --folder data_dir --batch_size 512
```

For training on GPU/s:
```bash
# 0 is the first GPU, 1 is the second, etc.
python train_finetune.py --folder data_dir --batch_size 512 --gpus 0, 
```

### Training with our DataModule 📉

As long as each of the image pairs have the same stem name (i.e. `coco_img1.png` and `coco_img1.txt`) all that you need to do is specify the folder on runtime. Any subfolder structure will be ignored, meaning `foo/bar/image1.jpg` will always find its `myster/folder/image1.txt` so long as they share a common parent folder. All image suffixes will work, the only expectation is that captions are separated by `\n`.

### Traing with you own Data 📊

If you have different training needs you may drop in your very own DataLoader. Edit the `train.py` script to you needs by commenting out our DataModule and inserting your own into `trainer.fit(model, your_data)`. The only expectation is that the first item of the return tuple is the image batch, and the second is the text batch.

## Goal ⚽

Our aim is to create an easy to use Lightning implementation of OpenAI's clip training script. We want our end product to be as inline with the orignal paper as possible. We will live by:

<p align="center">
    <img src="images/clip-paper.PNG" alt="CLIP Section Image">
</p>

## Citing

To cite this exact repo feel free to use:
```
@misc{cg2021trainCLIP,
  author = {Cade Gordon},
  title = {train-CLIP},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4915843},
  howpublished = {\url{https://github.com/Zasder3/train-CLIP}}
}
```

Learning transferable visual models from natural language supervision (a.k.a. CLIP)
```
@article{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
```

Data-Efficient Language-Supervised Zero-Shot Learning with Self-Distillation
```
@article{cheng2021data,
  title={Data-Efficient Language-Supervised Zero-Shot Learning with Self-Distillation},
  author={Cheng, Ruizhe and Wu, Bichen and Zhang, Peizhao and Vajda, Peter and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2104.08945},
  year={2021}
}
```

## TODO ✅

- [x] Get OpenAI's model creation script
- [x] Create model inits
  - [x] ResNet50
  - [x] ResNet50x4
  - [x] ResNet101
  - [x] ViT-B/32
  - [x] all models
- [x] Create model wrapper
- [x] Create lightning trainer
- [x] Create dataset files 
- [ ] Performance boosts
  - [x] Mixed-precision
  - [x] Self-distillation
  - [ ] Gradient checkpointing
  - [ ] Half-precision Adam statistics
  - [ ] Half-precision stochastically rounded text encoder weights
