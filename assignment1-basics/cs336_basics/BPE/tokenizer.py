import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.byte2id = {v:k for k, v in self.vocab.items()}
        """special key特殊处理"""
        if special_tokens:
            size = len(self.vocab)
            self.special_tok_id = {}
            for special_tok in special_tokens:
                special_tok = special_tok.encode('utf-8')
                if special_tok in self.byte2id:
                    self.special_tok_id[special_tok] = self.byte2id[special_tok]
                else:
                    self.vocab[size] = special_tok
                    self.special_tok_id[special_tok] = size
                    size += 1
    
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        """将vocab和merge转化为unicode编码"""
        import pickle

        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        
        normal_vocab: dict[int, bytes] = {}
        for key, context in vocab.items():
            children_key = int(key)
            if isinstance(context, str):
                children_context = context.encode("utf-8")
            normal_vocab[children_key] = children_context

        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        normal_merges = []
        for pair1, pair2 in merges:
            if isinstance(pair1, str):
                new_pair1 = pair1.encode("utf-8")
            if isinstance(pair2, str):
                new_pair2 = pair2.encode("utf-8")
            normal_merges.append((new_pair1, new_pair2))

        return cls(normal_vocab, normal_merges, special_tokens)
    
    """
    encode阶段主要分为两部分:
    1. pre-tokenize,将str分为一个个单词,返回一个列表
    2. merge, 将pre-tokenize返回的列表进行encode, 返回一个ID list
    """
    def pre_tokenize(self, text:str):
        pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        pre_list = []
        if not self.special_tokens:
            for match in re.finditer(pattern, text):
                text = match.group(0)
                pre_list.append(text)
            return pre_list

        self.special_tokens.sort(key=len, reverse=True)
        st = set(self.special_tokens)
        special_pattern = "|".join(re.escape(token) for token in self.special_tokens)    
        parts = re.split(f"({special_pattern})", text)
        for part in parts:
            if not part:
                continue
            if part in st:
                pre_list.append(part)
            else:
                for match in re.finditer(pattern, part):
                    text = match.group(0)
                    pre_list.append(text)
        
        return pre_list
    
    def apply_merge (self, word_embedding, pre_list) -> list[int]:
        id_list = []
        pre_list_idx = 0
        for word in pre_list:
            word_bytes = word_embedding[word]
            if self.special_tokens and word in self.special_tokens:
                word = word.encode('utf-8')
                tok_id = self.special_tok_id[word]
                id_list.append(tok_id)
                continue
            
            """对bytes序列进行apply_merge"""
            has_merged = True
            """
            忽略了次序
            
            while bytes_idx < len(word_bytes):
                if bytes_idx + 1 == len(word_bytes):
                    break
                elif has_merged:
                    bytes_idx = 0
                    has_merged = False
                else:
                    bytes1 = word_bytes[bytes_idx]
                    bytes2 = word_bytes[bytes_idx + 1]
                    if (bytes1, bytes2) in self.merges:
                        merge_byte = bytes1 + bytes2
                        word_bytes.pop(bytes_idx)
                        word_bytes.pop(bytes_idx)
                        word_bytes.insert(bytes_idx, merge_byte)
                        has_merged = True
                    else:
                        bytes_idx += 1
            """
            while has_merged:
                min_idx, max_pair = self.find_max_probality_pair(word_bytes)
                if not max_pair:
                    has_merged = False
                else:
                    idx = min_idx
                    bytes1 = word_bytes[idx]
                    bytes2 = word_bytes[idx + 1]
                    merge_bytes = bytes1 + bytes2
                    word_bytes.pop(idx)
                    word_bytes.pop(idx)
                    word_bytes.insert(idx, merge_bytes)
            
            for byte in word_bytes:
                bytes_id = self.byte2id[byte]
                id_list.append(bytes_id)
            
        return id_list
    
    def find_max_probality_pair(self, word_bytes):
        idx = 0
        min_idx = 0
        max_probality = len(self.merges)

        while idx < len(word_bytes)-1:
            bytes1 = word_bytes[idx]
            bytes2 = word_bytes[idx+1]
            if (bytes1, bytes2) in self.merges:
                pos = self.merges.index((bytes1, bytes2))
                """坐标越小，出现的概率越大"""
                if pos < max_probality:
                    max_probality = pos
                    min_idx = idx
            idx += 1
        
        if max_probality == len(self.merges):
            return min_idx, None
        return min_idx, self.merges[max_probality]


    def encode(self, text: str) -> list[int]:
        pre_list = self.pre_tokenize(text)
        word_embedding = {}
        for word in pre_list:
            word_embedding[word] = [bytes([b]) for b in word.encode('utf-8')]
        return self.apply_merge(word_embedding, pre_list)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        result = b""
        if not ids:
            return ""
        for id in ids:
            result += self.vocab[id]
        return result.decode("utf-8", errors = "replace") 
