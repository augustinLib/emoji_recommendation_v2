from torchtext.legacy import data

class DataLoader(object):

    def __init__(
        self, 
        train_filename,
        batch_size = 64,
        vaild_ratio = .2,
        device = -1,
        # 전체 단어 maximum 개수
        max_vocab = 999999,
        # 최소 몇번 나온얘들 추가할건지
        min_freq=1,
        # end_of_sentence 사용 여부
        use_eos = False,
        shuffle = True):

        super().__init__()
        
        #Field 정의
        self.label = data.Field(
            # class만 있기 때문에 non-sequential
            sequential = False,
            use_vocab = True,
            # 모르는 class가 있으면 안됨
            unk_token = False
        )

        self.text = data.Field(
            use_vocab = True,
            batch_first = True,
            include_lengths = False,
            eos_token = '<EOS>' if use_eos else None
        )

        # Dataset
        train, valid = data.TabularDataset(
            # 파일 불러오기
            path = train_filename,
            format = 'tsv',
            # 앞서 정의한 Field객체 추가
            fields = [
                ('label', self.label),
                ('text', self.text),
            ],
        #train_ratio 지정
        ).split(split_ratio=(1-vaild_ratio))

        # DataLoader
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
        # splits()을 이용해서 train_loader와 valid_loader를 각각 생성
        (train, valid),
        batch_size = batch_size,
        device='cuda:%d' % device if device >= 0 else 'cpu',
        shuffle=shuffle,
        # 문장 길이로 batch 형성
        sort_key=lambda x: len(x.text),
        # mini batch내에서 sorting 할거냐 (batch 안에서도 길이에 따라서 정렬)
        sort_within_batch=True,
        )

        # 단어사전 만들기
        # label에 대한 vocabulary
        self.label.build_vocab(train)
        # text에 대한 vocabulary
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)



