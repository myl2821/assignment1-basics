from collections import defaultdict
from collections.abc import Iterable, Iterator
import os
import regex as re
import multiprocessing as mp
from typing import BinaryIO

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int,bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @staticmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        raise NotImplementedError


    def encode(self, ids: list[int]) -> str:
        """Encode an input text into a sequence of token IDs."""
        return ""

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is required for memory-eﬀicient
        tokenization of large files that we cannot directly load into memory.
        """

        return []

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        return ""
        

class BPETrainer:

    @staticmethod
    def train(
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

        merges = []

        # Initialize vocab special tokens and single-byte tokens
        # follow the GPT-2 patterns that special tokens puts in the head of queue
        initial_tokens = [tok.encode("UTF-8") for tok in special_tokens] + [bytes([i]) for i in range(256)]
        vocab: list[bytes] = initial_tokens

        pre_tokenize_freqs: dict[bytes, int] = pre_tokenize(input_path, special_tokens)
        # tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], bytes]
        pair_freqs, pairs_to_keys = get_pair_freqs(pre_tokenize_freqs)

        for _ in range(len(vocab), vocab_size):
            # get token pair with max freq, and use the key itself as the tie-breaker
            subtoken_pair = max(pair_freqs, key=lambda k: (pair_freqs[k], k))
            subtoken1, subtoken2 = subtoken_pair
            merges.append(subtoken_pair)
            vocab.append(subtoken1+subtoken2)
            merge(pre_tokenize_freqs, pair_freqs, subtoken_pair, pairs_to_keys)

        vocab_dict = {i: v for i, v in enumerate(vocab)}
        return (vocab_dict, merges)

def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[tuple[bytes], dict[tuple[bytes], int]]:
    """
    Replaces every occurrence of pair=(x,y) in old_repr with the merged symbol x+y.
    The new freq is also calculated and returned.
    """
    new_symbols = []
    new_freq = defaultdict(int)
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbol = old_repr[i] + old_repr[i + 1]
            new_symbols.append(new_symbol)  # merges, e.g. b'A' + b'B' => b'AB'
            new_freq[new_symbol] 
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)

def merge(
        freqs: dict[tuple[bytes], int],
        pair_freqs: dict[tuple[bytes, bytes], int],
        subtoken_pair: tuple[bytes, bytes],
        pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]]):

    # get the key set that a subtoken pair shoule be merged
    keys = pairs_to_keys[subtoken_pair]

    for old_key in keys.copy():
        new_key = build_new_repr(old_key, subtoken_pair)
        old_freq = freqs.pop(old_key)
        freqs[new_key] += old_freq

        # Decrement frequencies in pair_freqs for old_key's adjacencies
        for i in range(len(old_key) - 1):
            left, right = old_key[i], old_key[i + 1]
            pair_freqs[(left, right)] -= old_freq
            if pair_freqs[(left, right)] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[(left, right)].discard(old_key)

        # Increment frequencies for new_key's adjacencies
        for i in range(len(new_key) - 1):
            left, right = new_key[i], new_key[i + 1]
            pair_freqs[(left, right)] += old_freq
            pairs_to_keys[(left, right)].add(new_key)
    
    pairs_to_keys[subtoken_pair] = set()


def pre_tokenize(
        input_path: str | os.PathLike,
        special_tokens: list[str],
    ) -> dict[bytes, int]:
    """
    Splits a file into chunks aligned with <|endoftext|>, then tokenizes each chunk
    in parallel. Returns aggregated frequency dict.
    """
    cpu_count = mp.cpu_count()
    pool = mp.Pool(processes=cpu_count)
    special_pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens)) if special_tokens else None
    special_token_bytes = list(map(lambda x: x.encode('UTF-8'), special_tokens))

    futures = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 2*cpu_count, list(map(lambda x: x.encode('UTF-8'), special_tokens)))
        assert(len(boundaries) >= 2)
        for i in range(len(boundaries)-1):
            boundary_start = boundaries[i]
            boundary_end = boundaries[i+1]
            f.seek(boundary_start)
            chunk_bytes = f.read(boundary_end - boundary_start)
            chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
            futures.append(pool.apply_async(pre_tokenize_chunk, (chunk_str, special_pattern)))

    pool.close()
    pool.join()

    freq_dicts = [future.get() for future in futures]
    agg_dict: dict[bytes, int] = defaultdict(int)
    for freq_dict in freq_dicts:
        for k, v in freq_dict.items():
            agg_dict[k] += v

    return agg_dict


def get_pair_freqs(
    freqs: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    """
    Builds a pair-frequency table and reverse mapping (pair -> set of keys).
    The reverse mapping will be used in the merging phase.
    """
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)

    for token, freq in freqs.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freqs[pair] += freq
            pairs_to_keys[pair].add(token)

    return pair_freqs, pairs_to_keys
    

# take from cs336_basics/pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    Returns:
        list[int]: the boundary of each chunk
    """

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

            # If there is any special token in the mini chunk?
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
            if found_at != -1:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# \p{L} matches a single code point in the unicode category "letter".
# \p{N} matches any kind of unicode numeric character in any script.
# ?[^\s\p{L}\p{N}]+ is used to match one or more occurrences of special characters, possibly preceded by a single special character
# \s+(?!\S) matches one or more whitespace characters that are not immediately followed by a non-whitespace character
GPT2_TOKENIZER_REGEX = \
    re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenize_chunk(chunk: str, special_pattern: re.Pattern | None) -> dict[tuple[bytes], int]:
    """Regex tokenizes the chunk. Splits first on special tokens, then uses PAT."""
    # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23

    freqs: dict[tuple[bytes], int] = defaultdict(int)
    chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for chunk in chunks:
        for match in GPT2_TOKENIZER_REGEX.finditer(chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] += 1

    return freqs