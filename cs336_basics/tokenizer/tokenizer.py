from collections.abc import Iterable, Iterator

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
        
