import torch
import torch.nn as nn

class SubjectClassifier(nn.modules):
  
    def __init__(self, input_size, output_size, use_batch_norm=True, dropout_p = .4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        self.block = nn.Sequential(
        nn.Linear(input_size, output_size)
        )