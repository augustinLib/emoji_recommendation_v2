import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from emoji_classification.trainer import Trainer
from emoji_classification.loader import DataLoader

from emoji_classification.models.model_sentiment import EmojiClassifier

# for CLI
'''
def define_hyperparameter():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required = True)
    p.add_argument('--train_fn', required = True)
    p.add_argument('--gpu_id', type = int, default = -1)
    p.add_argument('--verbose', type = int, default=2)

    p.add_argument('--min_vocab_freq', type = int, default = 5)
    p.add_argument('--max_vocab_size', type = int, default=999999)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type = int, default = 256)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--max_length', type=int, default=256)

    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    config = p.parse_args()

    return config
'''

# for colab
class Configure():
    def __init__(self,
                 model_fn,
                 train_fn,
                 gpu_id = -1,
                 verbose = 2,
                 min_vocab_freq = 5,
                 max_vocab_size = 999999,
                 batch_size = 512,
                 n_epochs = 50,
                 word_vec_size = 256,
                 dropout = 0.3,
                 max_length = 256,
                 hidden_size = 512,
                 n_layers = 4
                 ):
        
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.min_vocab_freq = min_vocab_freq
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.word_vec_size = word_vec_size
        self.dropout = dropout
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers




def main(config):
    loaders = DataLoader(
        train_filename=config.train_fn,
        batch_size = config.batch_size,
        device=config.gpu_id,
        max_vocab=config.max_vocab_size,
        min_freq=config.min_vocab_freq
    )
    # print sizes
    print(
        "|train| = ", len(loaders.train_loader.dataset),
        "|valid| = ", len(loaders.valid_loader.dataset)
    )
    vocab_size = len(loaders.text.vocab)
    num_classes = len(loaders.label.vocab)

    print("|vocab| = ", vocab_size, "|classes| = ", num_classes)


    # build model, optimizer, critic
    model = EmojiClassifier(
        input_size=vocab_size,
        word_vec_size=config.word_vec_size,
        hidden_size=config.hidden_size,
        num_class=num_classes,
        n_layers=config.n_layers,
        dropout_p=config.dropout
    )
    optimizer = optim.Adam(model.parameters())
    critic = nn.NLLLoss()
    print(model)

    # train
    # use gpu??
    if config.gpu_id >=0:
        model.cuda(config.gpu_id)
        critic.cuda(config.gpu_id)

    trainer = Trainer(config)
    trainer_model = trainer.train(
        model,
        critic,
        optimizer,
        loaders.train_loader,
        loaders.valid_loader
    )

    torch.save({
        'model' : trainer_model.state_dict(),
        'config' : config,
        'vocab' : loaders.text.vocab,
        'classes': loaders.label.vocab
    }, config.model_fn)



if __name__ == '__main__':
    # for terminel
    #config = define_hyperparameter()
    
    #for colab
    config = Configure(
        model_fn="./emoji_classification/models/epoch20_wordvec128.pth",
        train_fn="./data/sentimental.tok.bpe.tsv",
        gpu_id = 0,
        verbose = 2,
        min_vocab_freq = 5,
        max_vocab_size = 999999,
        batch_size = 256,
        n_epochs = 50,
        word_vec_size = 256,
        dropout = 0.3,
        max_length = 256,
        hidden_size = 512,
        n_layers = 4
    )
    main(config)