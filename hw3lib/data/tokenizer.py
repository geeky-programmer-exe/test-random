import os
from typing import Literal, List, Dict, Optional, Union
from tokenizers import Tokenizer, decoders, processors

class H3Tokenizer:
    """
    Tokenizer supporting character-level and subword tokenization strategies for ASR.
    
    Attributes:
        token_type (str): 'char', '1k', '5k', or '10k'.
        tokenizer (Tokenizer): HuggingFace tokenizers instance.
        vocab_size (int): Total vocabulary size.
        pad_id (int): ID for '[PAD]'.
        unk_id (int): ID for '[UNK]'.
        mask_id (int): ID for '[MASK]'.
        sos_id (int): ID for '[SOS]'.
        eos_id (int): ID for '[EOS]'.
        blank_id (int): ID for '[BLANK]'.
    """
    VALID_TYPES = ['char', '1k', '5k', '10k']

    def __init__(
        self,
        token_map: Optional[Dict[str, str]] = None,
        token_type: Literal['char', '1k', '5k', '10k'] = 'char',
        validate: bool = False
    ):
        if token_type not in self.VALID_TYPES:
            raise ValueError(f"token_type must be one of {self.VALID_TYPES}, got '{token_type}'")

        self.token_type = token_type
        
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(module_dir, 'tokenizer_jsons'),
            os.path.join(module_dir, '..', 'tokenizer_jsons'),
            os.path.join(os.path.dirname(module_dir), 'tokenizer_jsons')
        ]
        
        def _resolve_path(path: str) -> str:
            if os.path.exists(path):
                return path
            cand1 = os.path.join(module_dir, path)
            if os.path.exists(cand1):
                return cand1
            cand2 = os.path.join(module_dir, '..', path)
            if os.path.exists(cand2):
                return cand2
            fname = os.path.basename(path)
            for cd in candidate_dirs:
                cand3 = os.path.join(cd, fname)
                if os.path.exists(cand3):
                    return cand3
            return path

        if token_map is None:
            json_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
            if json_dir is None:
                raise FileNotFoundError("Could not locate 'tokenizer_jsons' directory.")
                
            token_map = {
                "char": os.path.join(json_dir, 'tokenizer_char.json'),
                "1k": os.path.join(json_dir, 'tokenizer_1000.json'),
                "5k": os.path.join(json_dir, 'tokenizer_5000.json'),
                "10k": os.path.join(json_dir, 'tokenizer_10000.json')
            }
        else:
            token_map = {k: _resolve_path(v) for k, v in token_map.items()}

        self.token_map = token_map
        self.tokenizer = Tokenizer.from_file(self.token_map[token_type])

        # Configure decoder based on strategy
        if self.token_type != 'char':
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            self.tokenizer.decoder = decoders.ByteLevel()
        else:
            self.tokenizer.decoder = decoders.Fuse()

        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.sos_id = self.tokenizer.token_to_id("[SOS]")
        self.eos_id = self.tokenizer.token_to_id("[EOS]")
        self.blank_id = self.tokenizer.token_to_id("[BLANK]")

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.decode_batch(batch_token_ids, skip_special_tokens=skip_special_tokens)

    def get_avg_chars_per_token(self, sample_texts: Optional[List[str]] = None) -> float:
        if sample_texts is None:
            sample_texts = [
                "the quick brown fox jumps over the lazy dog",
                "automatic speech recognition with transformer models"
            ]
        total_chars = sum(len(t) for t in sample_texts)
        total_tokens = sum(len(self.encode(t)) for t in sample_texts)
        return total_chars / max(total_tokens, 1)


# Compatibility alias
TextTokenizer = H3Tokenizer
