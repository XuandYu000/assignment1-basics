import os
import regex as re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO

MAX_BYTE_NUM = 256
ENCODE_TYPE = "utf-8"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_SPECIAL_TOKEN = "<|endoftext|>".encode(ENCODE_TYPE)

def init_vocab(special_tokens: list[str]):
    vocab = dict[int, bytes]()
    for i in range(MAX_BYTE_NUM):
        vocab[i] = bytes([i])
    
    for idx, token in enumerate(special_tokens):
        vocab[MAX_BYTE_NUM + idx] = token.encode(ENCODE_TYPE)
    
    return vocab

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(args: tuple[str, int, int, list[str]]):
    """Given a chunk(defined by inpu_path, start, end), pre-token this chunk
    Args:
        args (tuple[str, int, int, list[str]]):
            input_path: the file path
            start, end: the location of the start and end indexes of this chunk
            special_tokens: the special tokens removed before pre-tokenization
    
    Returns:
        list[list[bytes]]:
            pre-tokens. Each list item is a list of bytes. For example, "low" -> [b'l', b'o', b'w']
    """
    input_path, start, end, special_tokens = args

    pre_tokens: list[list(bytes)] = []
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(ENCODE_TYPE, errors="ignore")
    
    # 1. Remove special tokens
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens) # use re.escape since '|' may occur in the special tokens
    documents = re.split(special_token_pattern, chunk)

    # 2. pre-tokenization
    for doc in documents:
        tokens = [match.group(0).encode(ENCODE_TYPE) for match in re.finditer(PAT, doc)]
        for token in tokens:
            pre_tokens.append([bytes([item]) for item in token])
    
    return pre_tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = kwargs.get("num_processes", 16)

    vocab, merges = dict[int, bytes](), list[tuple[bytes, bytes]]()
    # 1. vocabulary initialization
    vocab = init_vocab(special_tokens)

    # 2. Pre-tokenization (Paralization)
    pre_token_args = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, SPLIT_SPECIAL_TOKEN)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            pre_token_args.append((input_path, start, end, special_tokens))
    
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(pre_tokenization, pre_token_args)
    
    # 3. Compute BPE merges
    pre_token_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    
    # merge: 一个pre_token相邻两个bytes， e.g. low -> b'lo', b'ow'
    # pair_counter: 计数每个merge
    # pair_indices: 倒排索引表，每个merge出现在那个pre_token中
    pair_counter: dict[tuple(bytes, bytes), int]=defaultdict(int)
    pair_indices: dict[tuple(bytes, bytes), set] = defaultdict(set)

    for idx, pre_token in enumerate(pre_token_bytes):
        for bytes1, bytes2 in zip(pre_token[:-1], pre_token[1:]):
            cand = (bytes1, bytes2)
            pair_counter[cand] += 1
            pair_indices[cand].add(idx)
    
    total_vocab: int = len(vocab)
    # merge step
    while total_vocab < vocab_size:
        if not pair_counter:
            break

        def find_max_pair(counter):
            max_pair, max_cnt = None, -1
            for pair, cnt in counter.items():
                if cnt > max_cnt:
                    max_pair, max_cnt = pair, cnt
                elif cnt == max_cnt:
                    if max_pair is None or pair > max_pair:
                        max_pair, max_cnt = pair, cnt # pair > max_pair select the one which is lexicographically larger
            return max_pair, max_cnt

        max_pair, max_cnt = find_max_pair(pair_counter)

        merge = max_pair
        new_token = merge[0] + merge[-1]

        # add new_token to vocab, add max_pair to merges
        vocab[total_vocab] = new_token
        merges.append(merge)

        # 修改pair_counter: merge[0], merge[-1]计数减一；new_token计数加一
        # 修改pair_indices: 含有merge[0], merge[-1]的位置注销，修改pre_token_bytes后重新注册到该位置
        affected_indices = pair_indices[merge].copy() # !!! import
        for indice in affected_indices:
            pre_token = pre_token_bytes[indice]

            # 整个list包含的candidate merge都要注销
            for bytes1, bytes2 in zip(pre_token[:-1], pre_token[1:]):
                cand = (bytes1, bytes2)
                pair_counter[cand] -= 1
                pair_indices[cand].discard(indice) # 注销倒排索引中当前indice
                if pair_counter[cand] == 0:
                    pair_counter.pop(cand)
                    pair_indices.pop(cand, None)
            
            idx = 0
            new_pre_token: list[bytes] = []
            while idx < len(pre_token):
                if idx < len(pre_token) - 1 and pre_token[idx] + pre_token[idx + 1] == new_token:
                    new_pre_token.append(new_token)
                    idx += 2
                else:
                    new_pre_token.append(pre_token[idx])
                    idx += 1

            pre_token_bytes[indice] = new_pre_token
            for bytes1, bytes2 in zip(new_pre_token[:-1], new_pre_token[1:]):
                cand = (bytes1, bytes2)
                pair_counter[cand] += 1
                pair_indices[cand].add(indice)

        total_vocab += 1

    return vocab, merges