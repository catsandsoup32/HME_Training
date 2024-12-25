import collections
import re
from typing import Dict, List, Tuple, Union


class LaTeXTokenizer:
    def __init__(self):
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        self.vocab = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text: str) -> List[str]:
        # Tokenize LaTeX using regex to capture commands, numbers and other characters
        return re.findall(r"\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S", text)

    def build_vocab(self, texts: List[str]):
        # Add special tokens to vocabulary
        for token in self.special_tokens:
            if token not in self.vocab: 
                self.vocab[token] = len(self.vocab)

        # Create a counter to hold token frequencies
        counter = collections.Counter()

        # Tokenize each text and update the counter
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Add tokens to vocab based on their frequency
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Build dictionaries for token to ID and ID to token conversion
        self.token_to_id = self.vocab
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}

    def encode(self, text: str) -> List[int]:
        # Tokenize the input text and add start and end tokens
        tokens = ["[BOS]"] + self.tokenize(text) + ["[EOS]"]

        # Map tokens to their IDs, using [UNK] for unknown tokens
        unk_id = self.token_to_id["[UNK]"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        # Map token IDs back to tokens
        tokens = [self.id_to_token.get(id, "[UNK]") for id in token_ids]
 
        # Remove tokens beyond the [EOS] token
        if "[EOS]" in tokens:
            tokens = tokens[: tokens.index("[EOS]")]

        # Replace [UNK] with ?
        tokens = ["?" if token == "[UNK]" else token for token in tokens]

        # Reconstruct the original text, ignoring special tokens
        return "".join([token for token in tokens if token not in self.special_tokens])