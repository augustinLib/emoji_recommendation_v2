import torch
import torch.nn as nn

class EmojiClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        num_class,
        n_layers=4,
        dropout_p=.3,
        # 몇 개의 단어를 보는 filter를 형성할 것인지
        filter_sizes = [3,4],
        # 몇 개의 패턴을 볼 것인지
        n_filters = [20, 20]
    ):
        # assign
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.filter_sizes = filter_sizes
        self.word_vec_size = word_vec_size
        self.n_filters = n_filters

        super().__init__()

        self.embedding = nn.Embedding(input_size, word_vec_size)

        # 긱긱의 지정한 filter끼리 모듈 생성
        self.pattern_extractor = nn.ModuleList()
        for filter_size, n_filter in zip(filter_sizes, n_filters):
            self.pattern_extractor.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = 1, out_channels = n_filter,
                        kernel_size = (filter_size, word_vec_size)
                    ),
                    #nn.LeakyReLU(),
                    nn.BatchNorm2d(n_filter)
                    
                )
            )
        
        
        self.rnn = nn.LSTM(
            # cnn 모델의 output input으로 추가
            input_size =1,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.Linear(hidden_size * 2, num_class),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # embedding layer에 input 통과
        x = self.embedding(x)

        # embedding layer값 cnn에 통과
        # 그 전에, 해야할 것들
        # 문장의 길이가 filter 크기보다 작을 경우 padding 해줘야함

        min_threshold = max(self.filter_sizes)
        if (min_threshold > x.size(1)):
            # padding tensor 생성
            pad = x.new(x.size(0), min_threshold - x.size(1), self.word_vec_size).zero_()
            # 원본과 padding tensor 병합
            x = torch.cat([x, pad], dim = 1)

        # CNN은 입력받을 때 4차원으로 입력받는다(batch_size, channel(filter 개수), height, width)
        # unsqueeze로() channel 부분 차원 추가해줌      
        x = x.unsqueeze(1)

        outputs_cnn = []
        for block in self.pattern_extractor:

            # window(filter)크기와 입력 크기가 같아서 마지막 차원은 1로 줄어듬
            # |output_cnn | = (batch_size, n_filter, (length - window_size + 1), 1)            
            output_cnn = block(x)

            output_cnn = nn.functional.max_pool1d(
                # 마지막 차원 제거
                input = output_cnn.squeeze(-1),
                kernel_size = output_cnn.size(-2)
            ).squeeze(-1)
            # |output_cnn| = (batch_size, n_filter, 1) -> (batch_size, n_filter)

            outputs_cnn += [output_cnn]
        
        outputs_cnn = torch.cat(outputs_cnn, dim=-1)
        # |output_cnn| = (batch_size, sum(n_filters))
        outputs_cnn = outputs_cnn.unsqueeze(dim=2)
        x, _ = self.rnn(outputs_cnn)
    
        # |x| = (batch_size, hidden_size * 2)

        x = x[:, -1]
        # x[:, -1] = (batch_size, hidden_size * 2)

        y = self.layers(x)
        # |y| = (batch_size, n_classes)

        return y

        
