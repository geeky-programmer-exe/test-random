import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp


class SearchTree:
    """Represents a deterministic tree for beam/greedy search testing"""
    def __init__(self, token_id: int, score: float = 0.0):
        self.token_id = token_id
        self.score    = score
        self.children = []
    
    def add_path(self, tokens: List[int], score: float):
        """Add a path to the tree with corresponding scores"""
        current = self
        for token in tokens:
            child = next((c for c in current.children if c.token_id == token), None)
            if not child:
                child = SearchTree(token)
                child.score = score
                current.children.append(child)
            current = child

    def pretty_print(self, tokenizer=None, depth=0, score=None):
        indent = "  " * depth
        token_str = str(self.token_id)
        if tokenizer:
            token_str = f"{self.token_id:<5} ({tokenizer.decode([self.token_id])})"
        
        score_str = f", score={score:>8.3f}" if score is not None else " " * 15
        print(f"{indent}{token_str}{score_str}")
        
        sorted_children = sorted(self.children, key=lambda x: -x.score)
        for child in sorted_children:
            child.pretty_print(tokenizer, depth + 1, child.score)


def make_tree(tokenizer, paths: List[Tuple[List[int], float]]):
    root = SearchTree(tokenizer.sos_id)
    for tokens, score in paths:
        root.add_path(tokens, score)
    return root 


class DeterministicScoreFn:
    """Score function using an explicit tree for beam/greedy search testing"""
    def __init__(self, trees: List[SearchTree], tokenizer):
        self.trees      = trees
        self.tokenizer  = tokenizer
        
    def __call__(self, x) -> jax.Array:
        x_np = np.asarray(x)
        batch_size = x_np.shape[0]
        seq_len    = x_np.shape[1]
        scores     = np.full((batch_size, seq_len, self.tokenizer.vocab_size), 1.0, dtype=np.float32)
        
        for batch_idx in range(batch_size):
            scores[batch_idx, -1, :] = self.update_scores(self.trees[batch_idx], x_np[batch_idx])
        
        last_step = jnp.asarray(scores[:, -1, :])
        return jax.nn.log_softmax(last_step, axis=-1)
    
    def update_scores_helper(self, root: SearchTree, x: np.ndarray, scores: np.ndarray):
        if x.shape[0] != 0 and root.token_id != int(x[0]):
            return
        
        if x.shape[0] == 0:
            scores[root.token_id] = root.score
            return
        
        if root.children:
            for child in root.children:
                self.update_scores_helper(child, x[1:], scores)
        else:
            scores[self.tokenizer.eos_id] = root.score
        return
    
    def update_scores(self, root: SearchTree, x: np.ndarray):
        scores = np.ones((self.tokenizer.vocab_size,), dtype=np.float32) 
        self.update_scores_helper(root, x, scores)
        return scores


def test_beam_search_single_batch(generator, tokenizer):
    """Test beam search single batch using an explicit tree structure"""
    print("Testing Single Batch Beam Search ...")

    paths = [
        [
            (tokenizer.encode("HELLO WORLD[EOS]"),  1000),
            (tokenizer.encode("YELLOW WORLD[EOS]"),  999),
            (tokenizer.encode("MELLOW WORLD[EOS]"),  998),
        ],
    ]
    
    max_len = max(len(tokens) for path in paths for tokens, score in path)
    beam_width = max(len(path) for path in paths)
    
    trees = [make_tree(tokenizer, path) for path in paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    gen = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max_len+10,
        device="cpu"
    )
    
    x = jnp.full((1, 1), tokenizer.sos_id, dtype=jnp.int32)
    seqs, scores = gen.generate_beam(x, beam_width=beam_width)
    
    expected_texts = ["HELLO WORLD", "YELLOW WORLD", "MELLOW WORLD"]
    for i, expected in enumerate(expected_texts):
        processed_seq = gen.post_process_sequence(seqs[0, i], tokenizer)
        generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
        print(f"Beam {i:<2} : generated: {generated:<12} | expected: {expected:<12}")
        assert generated == expected, f"Beam {i} generated '{generated}' but expected '{expected}'"


def test_beam_search_multi_batch(generator, tokenizer):
    """Test beam search using multiple explicit tree structures"""
    print("Testing Multi Batch Beam Search ...")
    
    batch_paths = [
        [
            (tokenizer.encode("HELLO WORLD[EOS]"),  1000),
            (tokenizer.encode("YELLOW WORLD[EOS]"), 999),
            (tokenizer.encode("MELLOW WORLD[EOS]"), 998)
        ],
        [
            (tokenizer.encode("GOOD BYE[EOS]"),  1000),
            (tokenizer.encode("GREAT DAY[EOS]"),  999),
            (tokenizer.encode("GUD NIGHT[EOS]"),  998)
        ],
    ]

    max_len = max(len(tokens) for batch in batch_paths for tokens, score in batch)
    beam_width = max(len(path) for path in batch_paths)
    
    trees = [make_tree(tokenizer, path) for path in batch_paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    gen = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max_len+10,
        device="cpu"
    )
    
    x = jnp.full((2, 1), tokenizer.sos_id, dtype=jnp.int32)
    seqs, scores = gen.generate_beam(x, beam_width=beam_width)
    
    expected_batch_texts = [
        ["HELLO WORLD", "YELLOW WORLD", "MELLOW WORLD"],
        ["GOOD BYE", "GREAT DAY", "GUD NIGHT"],
    ]
    
    for batch_idx, expected_texts in enumerate(expected_batch_texts):
        processed_seqs = gen.post_process_sequence(seqs[batch_idx], tokenizer)
        for beam_idx, (processed_seq, expected) in enumerate(zip(processed_seqs, expected_texts)):
            generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
            print(f"Batch {batch_idx:<2} : Beam {beam_idx:<2} : generated: {generated:<12} | expected: {expected:<12}")
            assert generated == expected, f"Batch {batch_idx}, Beam {beam_idx} generated '{generated}' but expected '{expected}'"


def test_greedy_search_single_batch(generator, tokenizer):
    """Test greedy search single batch"""
    print("Testing Single Batch Greedy Search ...")

    paths = [
        (tokenizer.encode("HELLO WORLD[EOS]"), 1000),
        (tokenizer.encode("YELLOW WORLD[EOS]"), 999),
        (tokenizer.encode("MELLOW WORLD[EOS]"), 998),
    ]

    tree = make_tree(tokenizer, paths)
    score_fn = DeterministicScoreFn([tree], tokenizer)

    gen = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=len(paths[0][0]),
        device="cpu"
    )

    x = jnp.full((1, 1), tokenizer.sos_id, dtype=jnp.int32)
    seq, _ = gen.generate_greedy(x)

    expected_text = "HELLO WORLD"
    processed_seq = gen.post_process_sequence(seq[0], tokenizer)
    generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
    print(f"Generated: {generated:<12} | Expected: {expected_text:<12}")
    assert generated == expected_text, f"Generated '{generated}' but expected '{expected_text}'"


def test_greedy_search_multi_batch(generator, tokenizer):
    """Test greedy search with multiple batches"""
    print("Testing Multi Batch Greedy Search ...")

    batch_paths = [
        [(tokenizer.encode("HELLO WORLD[EOS]"), 1000), (tokenizer.encode("YELLOW WORLD[EOS]"), 999), (tokenizer.encode("MELLOW WORLD[EOS]"), 998)],
        [(tokenizer.encode("GOOD BYE[EOS]"), 1000), (tokenizer.encode("GREAT DAY[EOS]"), 999), (tokenizer.encode("GUD NIGHT[EOS]"), 998)]
    ]

    trees = [make_tree(tokenizer, path) for path in batch_paths]
    score_fn = DeterministicScoreFn(trees, tokenizer)
    
    gen = generator(
        score_fn=score_fn,
        tokenizer=tokenizer,
        max_length=max(len(path[0][0]) for path in batch_paths),
        device="cpu"
    )

    x = jnp.full((2, 1), tokenizer.sos_id, dtype=jnp.int32)
    seq, _ = gen.generate_greedy(x)

    expected_texts = ["HELLO WORLD", "GOOD BYE"]
    for batch_idx, expected in enumerate(expected_texts):
        processed_seq = gen.post_process_sequence(seq[batch_idx], tokenizer)
        generated = tokenizer.decode(processed_seq.tolist(), skip_special_tokens=True)
        print(f"Batch {batch_idx:<2} : Generated: {generated:<12} | Expected: {expected:<12}")
        assert generated == expected, f"Batch {batch_idx} generated '{generated}' but expected '{expected}'"


def test_greedy_decoding(generator, tokenizer):
    """Run all greedy search tests"""
    test_greedy_search_single_batch(generator, tokenizer)
    test_greedy_search_multi_batch(generator, tokenizer)


def test_beam_decoding(generator, tokenizer):
    """Run all beam search tests"""
    test_beam_search_single_batch(generator, tokenizer)
    test_beam_search_multi_batch(generator, tokenizer)


def test_decoding(generator, tokenizer):
    """Run all decoding tests"""
    print("Testing Decoding ...")
    test_greedy_search_single_batch(generator, tokenizer)
    test_greedy_search_multi_batch(generator, tokenizer)
    test_beam_search_single_batch(generator, tokenizer)
    test_beam_search_multi_batch(generator, tokenizer)


def main():
    """Main function to run the tests"""
    from hw3lib.decoding import SequenceGenerator
    from hw3lib.data import H3Tokenizer
    from tests.testing_framework import TestingFramework

    parser = argparse.ArgumentParser(description='Run decoding tests')
    parser.add_argument('--mode', choices=['all', 'greedy', 'beam'], default='all',
                      help='Specify which decoding tests to run (default: all)')
    args, _ = parser.parse_known_args()

    dummy_config = {
        'tokenization': {
            'token_type': "1k",
            'token_map': {
                'char': './hw3lib/tokenizer_jsons/tokenizer_char.json',
                '1k'  : './hw3lib/tokenizer_jsons/tokenizer_1000.json',  
                '5k'  : './hw3lib/tokenizer_jsons/tokenizer_5000.json',
                '10k' : './hw3lib/tokenizer_jsons/tokenizer_10000.json'
            }
        },
    }
    
    tokenizer = H3Tokenizer(
        token_map  = dummy_config['tokenization']['token_map'], 
        token_type = dummy_config['tokenization']['token_type'], 
        validate   = False
    )

    test_categories = {
        'Decoding': []
    }

    if args.mode in ['all', 'greedy']:
        test_categories['Decoding'].append({
            'func': lambda: test_greedy_decoding(SequenceGenerator, tokenizer),
            'description': 'Test greedy decoding in JAX'
        })
    
    if args.mode in ['all', 'beam']:
        test_categories['Decoding'].append({
            'func': lambda: test_beam_decoding(SequenceGenerator, tokenizer),
            'description': 'Test beam decoding in JAX'
        })

    framework = TestingFramework(test_categories=test_categories)
    framework.run_tests()
    framework.summarize_results()


if __name__ == '__main__':
    main()
