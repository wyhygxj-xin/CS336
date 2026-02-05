import regex as re
from collections import defaultdict

class run_train_bpe:
    def __init__(self, input_file:str, vocab_size:int, special_tokens:list[str]):
        self.input_file = input_file
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.NBYTES = 256

    """ Pre_train的实现分为两部分:
    1. 读取文件内容并分块处理,格式转化为utf-8
    2. 对每个块进行Pre_train,构造最终的初始dict
    """
    def tackle_file(self, chunk_size:int = 1024 * 50, special_token:str = "<|endoftext|>"):
        """ 这一部分用来实现文件处理
            将文件分为一个个chunk, 并按次数返回给pre_train
            leftremain是用来记录chunk_size的剩余的
        """
        leftremain = ""
        special_token_len=len(special_token)
        with open(self.input_file, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                chunk = leftremain + chunk
                leftremain = ""
                special_token_pos = chunk.rfind(special_token)
                if special_token_pos == -1:
                    leftremain = chunk
                    continue
                chunk = chunk[:special_token_pos + special_token_len]
                leftremain = chunk[special_token_pos + special_token_len:]
                yield chunk
        if leftremain:
            yield leftremain
    
    def pre_train(self):
        word_counts = defaultdict(int)
        pattern = r"""(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        toks = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in self.special_tokens) 

        for chunk in self.tackle_file():
            blocks = re.split(special_pattern, chunk)
            for block in blocks:
                for match in re.finditer(pattern, block):
                    text = match.group(0)
                    word_counts[text] += 1

        return word_counts

    def train(self):
        """初始化vocab, mergelist"""
        vocab = {i: bytes([i]) for i in range(self.NBYTES)}
        for i, special_token in enumerate(self.special_tokens):
            vocab[self.NBYTES + i] = special_token.encode('utf-8')
        size = self.NBYTES + len(self.special_tokens)
        merge_list = []
        
        word_counts = self.pre_train()
        word_encoding = {}
        for word in word_counts:
            word_encoding[word] = list(word.encode('utf-8'))

        while size < self.vocab_size:
            max_pair = self.max_pair(word_counts, word_encoding)
            merge_list.append(max_pair)
            vocab[size] = max_pair
            size += 1
            self.merge(word_encoding, max_pair)
        
        return vocab, merge_list
    
    def max_pair(self, word_counts, word_encoding):
        pairs = defaultdict(int)
        for word, byte_list in word_encoding.items():
            count = word_counts[word]
            for i in range(len(byte_list)-1):
                pair = (byte_list[i], byte_list[i+1])
                pairs[pair] += count
        
        max_pair = ()
        max_count = 0
        for pair, count in pairs.items():
            if count == max_count:
                if pair[0] < max_pair[0]:
                    continue
                if pair[0] == max_pair[0] and pair[1] < max_pair[1]:
                    continue
            if count < max_count:
                continue
            max_pair = pair
            max_count = count
        
        return max_pair
    
    def merge(self, word_encoding, max_pair):
        for word, byte_list in word_encoding.items():
            print("test")
            i = 0
            new_tokens = []
            new_tokens_flag = False
            while i < len(byte_list):
            # for i in range(len(byte_list)-1):
                if i < len(byte_list) -1 and (byte_list[i], byte_list[i]+1) == max_pair:
                    new_tokens.append(max_pair[0]+max_pair[1])
                    new_tokens_flag = True
                    i += 2
                else:
                    new_tokens.append(byte_list[i])
                    i += 1

            if new_tokens_flag:
                print(new_tokens)
                word_encoding[word] = new_tokens
                

    



            


        
    

