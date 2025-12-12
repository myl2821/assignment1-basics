from collections.abc import Iterable, Iterator
import regex as re
import os
import multiprocessing as mp

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
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
        self.vocab_index = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_index = {pair: index for index, pair in enumerate(merges)}
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_token_pattern = re.compile("(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")")

            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_index:
                    self.vocab[next_id] = token_bytes
                    self.vocab_index[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = []
            self.special_token_pattern = None


    @staticmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        raise NotImplementedError


    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        chunks = re.split(self.special_token_pattern, text) if self.special_token_pattern else [text]
        
        output = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                output.append(self.vocab_index[chunk.encode('UTF-8')])
            else:
                tokens = pre_tokenize_chunk(chunk)
                for token in tokens:
                    output.extend(self._encode_token(token))

        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is required for memory-eﬀicient
        tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            for id in self.encode(text):
                yield(id)


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        out_buffer = b''
        for id in ids:
            subtoken = self.vocab[id]
            out_buffer += subtoken

        return out_buffer.decode('UTF-8', errors="replace")

    def _encode_token(self, token: tuple[bytes]) -> list[int]:
        token_list = list(token)
        while True:
            i = 0
            best_index = None
            best_rank = float("inf")
            while i < len(token_list)-1:
                subtoken_pair = (token_list[i], token_list[i+1])
                if subtoken_pair in self.merges_index:
                    rank = self.merges_index[subtoken_pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_index = i
                i += 1

            if best_index == None:
                break

            token_list = token_list[:best_index] + [token_list[best_index]+token_list[best_index+1]] + token_list[best_index+2:]
        
        return list(map(self.vocab_index.get, tuple(token_list)))

# \p{L} matches a single code point in the unicode category "letter".
# \p{N} matches any kind of unicode numeric character in any script.
# ?[^\s\p{L}\p{N}]+ is used to match one or more occurrences of special characters, possibly preceded by a single special character
# \s+(?!\S) matches one or more whitespace characters that are not immediately followed by a non-whitespace character
GPT2_TOKENIZER_REGEX = \
    re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenize_chunk(chunk: str) -> list[tuple[bytes]]:
    """Regex tokenizes the chunk. Splits first on special tokens, then uses PAT."""
    # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23

    tokens = []

    for match in GPT2_TOKENIZER_REGEX.finditer(chunk):
        match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
        tokens.append(match_bytes)

    return tokens