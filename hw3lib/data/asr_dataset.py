import os
from typing import Literal, Tuple, Optional, Union, List, Dict
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from .tokenizer import H3Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition) in JAX:

1. Data Organization:
   - Handles dataset partitions ('train-clean-100', 'dev-clean', 'test-clean')
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports normalization strategies:
     * global_mvn: Global mean and variance computed from training data
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training

3. Transcript Processing:
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H3Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
'''

class ASRDataset:
    def __init__(
        self,
        partition: str = 'train-clean-100',
        config: Optional[dict] = None,
        tokenizer: Optional[H3Tokenizer] = None,
        isTrainPartition: bool = True,
        global_stats: Optional[Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array]]] = None,
        root: Optional[str] = None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing in JAX.
        """
        # TODO: Implement __init__
        raise NotImplementedError # Remove once implemented

        if config is None:
            config = {
                'root': root or '',
                'num_feats': 80,
                'norm': 'cepstral',
                'specaug': False,
                'specaug_conf': {'time_mask_width_range': 10, 'freq_mask_width_range': 10}
            }

        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.train = isTrainPartition

        # Determine root directory
        data_root = self.config.get('root', root or '')
        if not data_root or not os.path.exists(data_root):
            # Check common workspace paths
            candidates = [
                os.path.join(os.getcwd(), 'hw3_data_subset', 'hw3p2_data'),
                os.path.join(os.getcwd(), '..', 'hw3_data_subset', 'hw3p2_data'),
                os.path.join(os.getcwd(), 'solutions', 'Jax', 'handout', 'hw3_data_subset', 'hw3p2_data'),
                os.path.join(os.getcwd(), 'solutions', 'Torch', 'handout', 'hw3_data_subset', 'hw3p2_data'),
                os.path.join(os.getcwd(), 'New folder', 'hw3p2_f26_torch', 'handout', 'hw3_data_subset', 'hw3p2_data'),
                'd:/Builds/IDL_F26/IDL-11785/F26_IDL_HW3P2/solutions/Jax/handout/hw3_data_subset/hw3p2_data',
                'd:/Builds/IDL_F26/IDL-11785/F26_IDL_HW3P2/New folder/hw3p2_f26_torch/handout/hw3_data_subset/hw3p2_data'
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    data_root = cand
                    break
        self.root = data_root

        if tokenizer is None:
            self.tokenizer = H3Tokenizer(token_type='char')
        else:
            self.tokenizer = tokenizer

        # TODO: Get tokenizer ids for special tokens (eos, sos, pad)
        self.eos_token = NotImplementedError
        self.sos_token = NotImplementedError
        self.pad_token = NotImplementedError

        # Directory paths
        self.partition_dir = os.path.join(self.root, partition)

        # TODO: Use root and partition to get the feature directory
        self.fbank_dir = NotImplementedError

        # TODO: Get all feature files in the feature directory in sorted order
        if os.path.exists(self.fbank_dir):
            self.fbank_files = NotImplementedError
        else:
            self.fbank_files = []

        # TODO: Take subset
        subset_size = NotImplementedError
        self.fbank_files = NotImplementedError

        # TODO: Get the number of samples in the dataset
        self.length = NotImplementedError

        # test-clean has no transcripts
        # TODO: Use root and partition to get the text directory
        self.text_dir = NotImplementedError
        if self.partition != 'test-clean' and os.path.exists(self.text_dir):
            # TODO: Get all text files in the text directory in sorted order
            self.text_files = NotImplementedError
            # TODO: Take subset
            self.text_files = NotImplementedError

            # Verify data alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")
        else:
            self.text_files = []

        # Initialize lists to store features and transcripts
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        # Initialize counters for character and token counts
        # DO NOT MODIFY
        self.total_chars = 0
        self.total_tokens = 0

        # Initialize max length variables
        # DO NOT MODIFY
        self.feat_max_len = 0
        self.text_max_len = 0

        num_feats = self.config.get('num_feats', 80)
        norm_type = self.config.get('norm', 'cepstral')

        # When norm is global_mvn and global_stats is None, update Welford
        # accumulators (count, mean, M2) in numpy. Otherwise store the provided stats.
        # DO NOT MODIFY
        self.global_mean = None
        self.global_std = None
        if norm_type == 'global_mvn':
            if global_stats is not None:
                self.global_mean = np.asarray(global_stats[0], dtype=np.float32)
                self.global_std = np.asarray(global_stats[1], dtype=np.float32)
            else:
                if not isTrainPartition:
                    raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
                count = 0
                mean = np.zeros(num_feats, dtype=np.float64)
                M2 = np.zeros(num_feats, dtype=np.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            # TODO: Load features
            # Features are of shape (num_feats, time)
            feat = NotImplementedError

            # TODO: Truncate features to num_feats set by you in the config
            feat = NotImplementedError

            # Append to self.feats (num_feats is set by you in the config)
            self.feats.append(feat)

            # Track max length (time dimension)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update Welford statistics (DO NOT MODIFY)
            if norm_type == 'global_mvn' and global_stats is None:
                batch_count = feat.shape[1]
                count += batch_count
                delta = feat - mean[:, None]
                mean += delta.mean(axis=1)
                delta2 = feat - mean[:, None]
                M2 += (delta * delta2).sum(axis=1)

            if self.partition != 'test-clean' and i < len(self.text_files):
                text_path = os.path.join(self.text_dir, self.text_files[i])
                # TODO: Load the transcript
                transcript = NotImplementedError

                # TODO: Track character count (before tokenization)
                # DO NOT MODIFY
                self.total_chars += len(transcript)

                # TODO: Use tokenizer to encode the transcript (see tokenizer.encode for details)
                tokens = NotImplementedError

                # Track token count (excluding special tokens)
                # DO NOT MODIFY
                self.total_tokens += len(tokens)

                # Track max length (add 1 for the sos/eos tokens)
                # DO NOT MODIFY
                self.text_max_len = max(self.text_max_len, len(tokens) + 1)

                # TODO: Create shifted and golden versions by adding sos and eos tokens
                t_shifted = NotImplementedError
                t_golden = NotImplementedError
                self.transcripts_shifted.append(t_shifted)
                self.transcripts_golden.append(t_golden)

        if self.partition != 'test-clean':
            # Verify data alignment
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # TODO: Compute final global_mean and global_std when using global_mvn
        if norm_type == 'global_mvn' and global_stats is None:
            variance = NotImplementedError
            self.global_std = NotImplementedError
            self.global_mean = NotImplementedError

        # Calculate average characters per token
        # DO NOT MODIFY
        self.avg_chars_per_token = self.total_chars / max(self.total_tokens, 1)

    def get_avg_chars_per_token(self) -> float:
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        return self.length

    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.
        Returns:
            Tuple: (feat, shifted_transcript, golden_transcript)
        """
        # TODO: Implement __getitem__
        raise NotImplementedError # Remove once implemented

        # TODO: Load features for this index as a float numpy array, shape (num_feats, time)
        feat = NotImplementedError
        norm_type = self.config.get('norm', 'cepstral')

        # TODO: Apply normalization
        # Apply normalization: (num_feats, time)
        if norm_type == 'global_mvn':
            assert self.global_mean is not None and self.global_std is not None, "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean[:, None]) / (self.global_std[:, None] + 1e-8)
        elif norm_type == 'cepstral':
            mean = feat.mean(axis=1, keepdims=True)
            std = feat.std(axis=1, keepdims=True) + 1e-8
            feat = (feat - mean) / std
        elif norm_type == 'none':
            pass

        # TODO: Get transcripts for non-test partitions
        if self.partition == 'test-clean':
            shifted = None
            golden = None
        else:
            shifted = NotImplementedError
            golden = NotImplementedError

        return feat, shifted, golden

    def collate_fn(self, batch: List[Tuple]):
        """
        Pads features and transcripts in the batch.

        Returns:
            Tuple: (batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lens, transcript_lens)
            where batch_feats_pad has shape (batch_size, max_feat_len, num_feats).
        """
        # TODO: Implement collate_fn
        raise NotImplementedError # Remove once implemented

        batch_size = len(batch)
        # TODO: Collect features from the batch. Each item is (num_feats, time);
        # the padded batch is (batch, max_time, num_feats), so transpose time to axis 1.
        feats = NotImplementedError
        num_feats = feats[0].shape[0]

        # TODO: Collect feature lengths from the batch
        feat_lens = NotImplementedError
        max_feat_len = int(np.max(feat_lens))

        # TODO: Pad features to create a batch of fixed-length padded features
        batch_feats_pad = NotImplementedError
        for i, f in enumerate(feats):
            batch_feats_pad[i, :f.shape[1], :] = NotImplementedError

        # TODO: Apply SpecAugment for training when config["specaug"] and isTrainPartition.
        # Features stay (batch, time, num_feats). Do not permute to a torch layout.
        if self.config.get("specaug", False) and self.isTrainPartition:
            conf = self.config.get("specaug_conf", {})
            # TODO: Apply frequency masking
            if conf.get("apply_freq_mask", False):
                num_mask = NotImplementedError
                f_max = NotImplementedError
                for _ in range(num_mask):
                    f = NotImplementedError
                    f0 = NotImplementedError
                    batch_feats_pad[:, :, f0:f0 + f] = NotImplementedError

            # TODO: Apply time masking
            if conf.get("apply_time_mask", False):
                num_mask = NotImplementedError
                t_max = NotImplementedError
                for _ in range(num_mask):
                    t = NotImplementedError
                    t0 = NotImplementedError
                    batch_feats_pad[:, t0:t0 + t, :] = NotImplementedError

        # TODO: Handle transcripts for non-test partitions
        if self.partition == 'test-clean':
            batch_shifted_pad = None
            batch_golden_pad = None
            transcript_lens = None
        else:
            shifteds = NotImplementedError
            goldens = NotImplementedError
            transcript_lens = NotImplementedError
            max_text_len = int(np.max(transcript_lens))

            batch_shifted_pad = NotImplementedError
            batch_golden_pad = NotImplementedError

            for i in range(batch_size):
                batch_shifted_pad[i, :len(shifteds[i])] = NotImplementedError
                batch_golden_pad[i, :len(goldens[i])] = NotImplementedError

        # TODO: Return padded features, padded shifted, padded golden, feature lengths, and transcript lengths
        return batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lens, transcript_lens


# Compatibility alias
ASRDataSource = ASRDataset
