import os
import regex as re
from typing import Iterable, Iterator
from .utils import PAT, ENCODE_TYPE

class Tokenizer:
    """
    Construct a tokenizer from a given vocabulary, list of merges,
    and (optionally) a list of special tokens.
    """
    def __init__(self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None
    ):
        self.vocab = vocab # id to bytes
        self.merges = merges
        self.special_tokens = special_tokens

        self.bytes2id = {v: k for k, v in self.vocab.items()}
        self.merge2id = {merge: idx for idx, merge in enumerate(merges)}

    def from_files(cls,
        vocab_filepath: str,
        merges_filepath: str,
        speical_tokens: list[str]|None = None
    ):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary 
        and list of merges (in the same format that your BPE training code output) 
        and (optionally) a list of special tokens.
        """
        vocab: dict[int, bytes] = ()
        merges: list[tuple[bytes, bytes]] = []

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab = json.load(f)
        
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        token_ids: list[int] = []

        # 1. Pre-tokenize
        def _pre_tokenize(text: str) -> list[list[bytes]]:
            """Given an input text, pre-tokenize the text.
            For example, "low" -> [b'l', b'o', b'w']
            Args:
                text (str): the input text
            
            Returns:
                list[list[bytes]]:
                pre-tokens. Each list item is a list of bytes.
            """
            pre_tokens: list[list[bytes]] = []

            special_tokens = self.special_tokens or []
            if special_tokens:
                # Longest specials first so a shorter token cannot steal a prefix of a longer one.
                ordered = sorted(special_tokens, key=len, reverse=True)
                special_token_pattern = "|".join(re.escape(t) for t in ordered)
                special_set = set(special_tokens)
                # 捕获组是正则表达式模式中被圆括号包裹的部分。
                # 当我们在re.split()方法中使用捕获组时，模式匹配到的内容会和字符串其余部分一同出现在最终结果里。
                parts = re.split(f"({special_token_pattern})", text)
            else:
                parts = [text]
                special_set = set()

            for part in parts:
                if part == "":
                    continue
                if part in special_set:
                    pre_tokens.append([part.encode(ENCODE_TYPE)])
                else:
                    for match in re.finditer(PAT, part):
                        token = match.group(0).encode(ENCODE_TYPE)
                        pre_tokens.append([bytes([item]) for item in token])

            return pre_tokens
        
        pre_tokens = _pre_tokenize(text)

        # 2. Apply the merges
        def _single_token_merge(pre_token: list[bytes]) -> list[int]:
            """Apply the vocabulary element merges created during BPE training to the pre_token in the same order of creation.
            Args:
                pre_token: A list of bytes represents the pre_token.
            
            Returns:
                merged: list[int]: a list of token ids.
            """
            merged: list[int] = []

            while True:
                if len(pre_token) < 2:
                    break
                
                all_merges = {idx: (bytes1, bytes2) for idx, (bytes1, bytes2) in enumerate(zip(pre_token[:-1], pre_token[1:])) if (bytes1, bytes2) in self.merges}
                
                if not all_merges:
                    break

                # find the merge with the smallest merge rank
                min_rank_merge = None
                min_rank_merge_idx = -1
                min_rank = float('inf')
                for idx, (bytes1, bytes2) in all_merges.items():
                    rank = self.merge2id[(bytes1, bytes2)]
                    if rank < min_rank:
                        min_rank = rank
                        min_rank_merge = (bytes1, bytes2)
                        min_rank_merge_idx = idx

                pre_token = pre_token[:min_rank_merge_idx] + [min_rank_merge[0] + min_rank_merge[1]] + pre_token[min_rank_merge_idx + 2:]
            
            merged = [self.bytes2id[token] for token in pre_token]
            return merged

        for pre_token in pre_tokens:
            token_ids += _single_token_merge(pre_token)

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of input texts into a sequence of token IDs.
        Args:
            iterable (Iterable[str]): an iterable of input texts.
        
        Returns:
            Iterator[int]: an iterator of token ids.
        """
        # 注意from的使用: 如果有多个text，一个text返回多个token时,
        # e.g. [text1: [token1, token2, token3], text2:[token4, token5], ...]
        # 期望迭代器迭代完全结果[token1, token2, token3, token4, token5, ...]
        # yield self.encode(text)将返回一个text的完整结果，迭代完全结果[[token1, token2, token3], [token4, token5], ...]
        # yield from self.encode(text)返回[token1, token2, token3, token4, token5, ...]
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of token IDs into a string.
        Args:
            token_ids (list[int]): a list of token ids.
        
        Returns:
            str: the decoded string.
        """
        decoded_bytes: bytes = b""
        for token_id in token_ids:
            decoded_bytes += self.vocab[token_id]
        return decoded_bytes.decode(ENCODE_TYPE, errors="replace")