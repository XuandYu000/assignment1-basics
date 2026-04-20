MAX_BYTE_NUM = 256
ENCODE_TYPE = "utf-8"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_SPECIAL_TOKEN = "<|endoftext|>".encode(ENCODE_TYPE)