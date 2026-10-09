from typing import Literal, Optional
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    from tests.testing_framework import TestingFramework
except ImportError:
    from testing_framework import TestingFramework


def test_dataset_asr(dataset, method: Optional[Literal['__init__', '__getitem__', 'collate_fn']] = None):
    """
    Collection of test cases for the JAX ASR dataset.
    """ 
    valid_args = ['__init__', '__getitem__', 'collate_fn']
    
    if method:
        if method not in valid_args:
            raise ValueError(f"Invalid method: {method}. Valid methods are {valid_args}.")
        
        if method == 'collate_fn':
            test_asr_data_collate_fn(dataset=dataset)
        elif method == '__getitem__':
            test_asr_data_getitem(dataset=dataset)
        elif method == '__init__':
            test_asr_data_init(dataset=dataset)
    else:
        test_asr_data_init(dataset=dataset)
        test_asr_data_getitem(dataset=dataset)
        test_asr_data_collate_fn(dataset=dataset)


def test_asr_data_init(dataset):
    """
    Test case to validate the __init__ method of the ASRDataset in JAX.
    """
    print("Testing __init__ method ...")

    # Assert dataset length matches FBANK files
    assert len(dataset) == len(dataset.fbank_files), "Dataset length mismatch with FBANK files."
    print("Test Passed: Dataset length matches FBANK files.")

    # Assert dataset length matches TRANSCRIPT files
    if dataset.partition != 'test-clean':
        assert len(dataset) == len(dataset.text_files), "Dataset length mismatch with TRANSCRIPT files."
        print("Test Passed: Dataset length matches TRANSCRIPT files.")

        # Assert order alignment between the FBANK files and TRANSCRIPT files
        for fbank_file, text_file in zip(dataset.fbank_files, dataset.text_files):
            f_base = os.path.splitext(fbank_file)[0]
            t_base = os.path.splitext(text_file)[0]
            assert f_base == t_base, \
                f"FBANK file {fbank_file} and TRANSCRIPT file {text_file} are misaligned."
        print("Test Passed: Order alignment between FBANK files and TRANSCRIPT files is correct.")
    
        # Assert alignment between features and transcripts
        assert len(dataset.feats) == len(dataset.transcripts_shifted) == len(dataset.transcripts_golden), \
            "Feature and transcript arrays are not aligned in length."
        print("Test Passed: Alignment between features and transcripts is correct.")
        
    # Validate num_feats
    for feat in dataset.feats:
        assert feat.shape[0] == dataset.config['num_feats'], \
            f"Feature mismatch: Expected {dataset.config['num_feats']} features, but got {feat.shape[0]}."
    print("Test Passed: All features have the correct number of dimensions (num_feats).")
        
    # Verify transcript decoding
    if dataset.partition != 'test-clean':
        for shifted, golden in zip(dataset.transcripts_shifted, dataset.transcripts_golden):
            s_list = shifted[1:].tolist() if hasattr(shifted, 'tolist') else list(shifted[1:])
            g_list = golden[:-1].tolist() if hasattr(golden, 'tolist') else list(golden[:-1])
            shifted_text = dataset.tokenizer.decode(s_list)
            golden_text = dataset.tokenizer.decode(g_list)
            assert golden_text == shifted_text, \
                f"Decoded transcript mismatch: {shifted_text} != {golden_text}"
        print("Test Passed: All transcripts are decoded correctly after removing SOS and EOS tokens.")


def test_asr_data_getitem(dataset):
    """
    Test case to validate the __getitem__ method of the ASRDataset in JAX.
    """
    print("Testing __getitem__ method ...")
    
    for idx in range(min(len(dataset), 5)):
        feat, shifted_transcript, golden_transcript = dataset[idx]
        
        # Assert feature is a float numpy/jax array
        assert isinstance(feat, (np.ndarray, jax.Array)), f"Feature is not a numpy/jax array, got {type(feat)}"
        assert feat.dtype in (np.float32, np.float64, jnp.float32), f"Feature dtype unexpected: {feat.dtype}"
        
        # Validate feature dimensions
        assert feat.shape[0] == dataset.config['num_feats'], \
            f"Feature mismatch: Expected {dataset.config['num_feats']} features, but got {feat.shape[0]}."
        
        # Check transcript behavior based on partition
        if dataset.partition == 'test-clean':
            assert shifted_transcript is None, f"Shifted transcript is not None for 'test-clean' at index {idx}."
            assert golden_transcript is None, f"Golden transcript is not None for 'test-clean' at index {idx}."
            print(f"Test Passed: Transcripts are None for 'test-clean' at index {idx}.")
        else:
            assert isinstance(shifted_transcript, (np.ndarray, jax.Array)), "Shifted transcript is not an array"
            assert isinstance(golden_transcript, (np.ndarray, jax.Array)), "Golden transcript is not an array"
            
            assert shifted_transcript.ndim == 1, "Shifted transcript is not of shape (seq_len,)"
            assert golden_transcript.ndim == 1, "Golden transcript is not of shape (seq_len,)"
            
            assert int(shifted_transcript[0]) == dataset.sos_token, f"Shifted transcript missing SOS token at index {idx}."
            assert int(golden_transcript[-1]) == dataset.eos_token, f"Golden transcript missing EOS token at index {idx}."
            
            s_list = shifted_transcript[1:].tolist() if hasattr(shifted_transcript, 'tolist') else list(shifted_transcript[1:])
            g_list = golden_transcript[:-1].tolist() if hasattr(golden_transcript, 'tolist') else list(golden_transcript[:-1])
            shifted_decoded = dataset.tokenizer.decode(s_list)
            golden_decoded = dataset.tokenizer.decode(g_list)
            assert shifted_decoded == golden_decoded, \
                f"Decoded mismatch for sample {idx}: {shifted_decoded} != {golden_decoded}"
            
    print("Test Passed: All samples have correct feature dimensions and transcript alignment.")


def test_asr_data_collate_fn(dataset):
    """
    Test case to validate the collate_fn method of the ASRDataset in JAX.
    """
    print("Testing collate_fn method ...")
    
    batch = [dataset[idx] for idx in range(min(len(dataset), 4))]
    batch_output = dataset.collate_fn(batch)
    
    batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lengths, transcript_lengths = batch_output
    
    assert isinstance(batch_feats_pad, (np.ndarray, jax.Array)), "Batch features are not array"
    assert batch_feats_pad.ndim == 3, \
        f"Feature batch has incorrect dimensions: Expected 3D array but got {batch_feats_pad.ndim}D array."
    print("Test Passed: Feature batch has correct dimensions (3D array).")
    
    max_feat_length = int(np.max(feat_lengths))
    assert batch_feats_pad.shape[1] == max_feat_length, "Inconsistent feature padding in batch."
    print("Test Passed: All sequences are padded to the same length.")

    if dataset.partition == 'test-clean':
        assert batch_shifted_pad is None, "Shifted transcripts batch is not None for 'test-clean'."
        assert batch_golden_pad is None, "Golden transcripts batch is not None for 'test-clean'."
        assert transcript_lengths is None, "Transcript lengths are not None for 'test-clean'."
        print("Test Passed: Transcripts and lengths are correctly set to None for 'test-clean'.")
    else:
        assert isinstance(batch_shifted_pad, (np.ndarray, jax.Array)), "Batch shifted transcripts not array"
        assert isinstance(batch_golden_pad, (np.ndarray, jax.Array)), "Batch golden transcripts not array"
        
        max_transcript_length = int(np.max(transcript_lengths))
        assert batch_shifted_pad.shape[1] == max_transcript_length, "Inconsistent transcript padding in batch."
        print("Test Passed: All transcripts are padded to the same length.")

        padding_value = dataset.pad_token
        for transcript, length in zip(batch_shifted_pad, transcript_lengths):
            assert all(int(x) == padding_value for x in transcript[int(length):]), "Padding values are incorrect."
        print("Test Passed: Padding values are correct.")


def main():
    """
    Main function to run the ASR dataset tests using the testing framework.
    """
    from hw3lib.data import ASRDataset, H3Tokenizer

    dummy_config = {
        'tokenization': {
            'token_type': "1k",
            'token_map': {
                'char': './hw3lib/data/tokenizer_jsons/tokenizer_char.json',
                '1k': './hw3lib/data/tokenizer_jsons/tokenizer_1000.json',
                '5k': './hw3lib/data/tokenizer_jsons/tokenizer_5000.json',
                '10k': './hw3lib/data/tokenizer_jsons/tokenizer_10000.json'
            }
        },
        'data': {
            'root': "./solutions/Jax/handout/hw3_data_subset/hw3p2_data",
            'subset': 1.0,
            'batch_size': 8,
            'NUM_WORKERS': 2,
            'num_feats': 80,
            'norm': 'global_mvn',
            'specaug': True,
            'specaug_conf': {
                'apply_time_mask': True,
                'apply_freq_mask': True,
                'num_freq_mask': 2,
                'num_time_mask': 2,
                'freq_mask_width_range': 10,
                'time_mask_width_range': 10
            }
        }
    }

    tokenizer = H3Tokenizer(
        token_map=dummy_config['tokenization']['token_map'],
        token_type=dummy_config['tokenization']['token_type'],
        validate=False
    )

    train_dataset = ASRDataset(
        partition='train-clean-100',
        config=dummy_config['data'],
        tokenizer=tokenizer,
        isTrainPartition=True,
        global_stats=None
    )

    global_stats = None
    if dummy_config['data']['norm'] == 'global_mvn':
        global_stats = (train_dataset.global_mean, train_dataset.global_std)

    test_dataset = ASRDataset(
        partition='test-clean',
        config=dummy_config['data'],
        tokenizer=tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )

    framework = TestingFramework(
        test_categories={
            'ASRDataset Train': [
                {
                    'func': lambda: test_dataset_asr(train_dataset),
                    'description': 'Test a Train instance of ASRDataset class in JAX'
                }
            ],
            'ASRDataset Test': [
                {
                    'func': lambda: test_dataset_asr(test_dataset),
                    'description': 'Test a Test instance of ASRDataset class in JAX'
                }
            ]
        }
    )

    framework.run_tests()
    framework.summarize_results()


if __name__ == "__main__":
    main()
