import os
import subprocess
import argparse
import torch
import torch.nn as nn

from torchtext.legacy import data 
from emoji_classification.models.model_sentiment import EmojiClassifier
from emoji.emojiPrint import return_image


def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    # 1개뿐 아니라 더 많은 클래스의 후보들을 출력할 수 있음
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)

    config = p.parse_args()

    return config


def read_text(max_length=256):
    lines = []
    line = input()
    line = os.popen('echo '+ line + ' | mecab -O wakati ' )
    line = line.read()
    if line.strip() != '':
        lines += [line.strip().split(' ')[:max_length]]


    return lines


def define_field():
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
        ),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None,
        )
    )


def main(config):
    saved_data = torch.load(
        "./final3.pth",
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    saved_model = saved_data['model']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=config.max_length)

    with torch.no_grad():
        model = EmojiClassifier(
            input_size=vocab_size,
            word_vec_size=train_config.word_vec_size,
            hidden_size=train_config.hidden_size,
            num_class=n_classes,
            n_layers=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
        model.load_state_dict(saved_model)



        y_hats = []

        model.eval()

        y_hat = []
        for idx in range(0, len(lines), config.batch_size):                
            x = text_field.numericalize(
                text_field.pad(lines[idx:idx + config.batch_size]),
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu',
            )

            y_hat += [model(x).cpu()]
        y_hat = torch.cat(y_hat, dim=0)

        y_hats += [y_hat]

        model.cpu()
        y_hats = torch.stack(y_hats).exp()


        probs, indice = y_hats.topk(config.top_k)

        for i in range(len(lines)):
            i = classes.itos[indice[i]]
            return_image(i)


if __name__ == '__main__':
    config = define_argparser()
    main(config)