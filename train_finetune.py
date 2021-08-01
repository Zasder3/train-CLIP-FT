import torch
import clip
import copy
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel


def main(hparams):
    clp, preprocess = clip.load("ViT-B/16", device='cpu')

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size
    
    for p in clp.parameters(): 
        p.data = p.data.float() 
        if p.grad:
            p.grad.data = p.grad.data.float()

    model = CustomCLIPWrapper(clp.visual, clp.transformer, hparams.minibatch_size, avg_word_embs=True)

    model.model.token_embedding = clp.token_embedding
    model.model.ln_final = clp.ln_final
    model.model.text_projection = clp.text_projection
    model.teacher = copy.deepcopy(model.model)

    dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
