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
                block = f.read(chunk_size)
                if not block:
                    break
                block = leftremain + block
                leftremain = ""
                special_token_pos = block.rfind(special_token)
                if special_token_pos == -1:
                    leftremain = block
                    continue
                chunk = block[:special_token_pos + special_token_len]
                leftremain = block[special_token_pos + special_token_len:]
                yield chunk
        if leftremain:
            yield leftremain
    
    def pre_train(self):
        word_counts = defaultdict(int)
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") 
        toks = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in self.special_tokens) 

        for chunk in self.tackle_file():
            blocks = re.split(special_pattern, chunk)
            print(blocks)
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
            max_pair = self.max_pair(word_counts, word_encoding, vocab)
            merge_pair = (vocab[max_pair[0]], vocab[max_pair[1]])
            merge_list.append(merge_pair)
            new_token = vocab[max_pair[0]]+ vocab[max_pair[1]]
            vocab[size] = new_token
            self.merge(word_encoding, max_pair, size)
            size += 1
        return vocab, merge_list
    
    def max_pair(self, word_counts, word_encoding, vocab):
        pairs = defaultdict(int)
        for word, byte_list in word_encoding.items():
            count = word_counts[word]
            for i in range(len(byte_list)-1):
                pair = (byte_list[i], byte_list[i+1])
                pairs[pair] += count
        
        max_pair = ()
        max_count = 0
        for pair, pair_count in pairs.items():
            if pair_count > max_count:
                max_pair = pair
                max_count = pair_count
            elif pair_count == max_count:
                pair_str = (vocab[pair[0]], vocab[pair[1]])
                max_pair_str = (vocab[max_pair[0]], vocab[max_pair[1]])
                if pair_str > max_pair_str:
                    max_pair = pair
                    max_count = pair_count      
        return max_pair
    
    def merge(self, word_encoding, max_pair, size):
        for word, byte_list in word_encoding.items():
            i = 0
            new_tokens = []
            new_tokens_flag = False
            while i < len(byte_list):
                
            # for i in range(len(byte_list)-1):
                if i < len(byte_list) -1 and (byte_list[i], byte_list[i+1]) == max_pair:
                    # 这里其实要把vocab传入做引导转换，append要引入新的ID值了
                    new_tokens.append(size)
                    new_tokens_flag = True
                    i += 2
                else:
                    new_tokens.append(byte_list[i])
                    i += 1

            if new_tokens_flag:
                word_encoding[word] = new_tokens
                

    



            


        
    

