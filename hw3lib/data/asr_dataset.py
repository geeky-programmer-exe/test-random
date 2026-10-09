import os
from typing import Literal, Tuple, Optional, Union, List, Dict
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from .tokenizer import H3Tokenizer

try:
    import grain.python as grain
    HAS_GRAIN = True
except ImportError:
    HAS_GRAIN = False


class ASRDataset:
    """
    ASR Dataset for loading and preprocessing speech filterbanks and transcripts in JAX.
    
    Specification:
    - Loads paired .npy speech features and .npy transcript files.
    - Supports normalizations: 'global_mvn', 'cepstral', 'none'.
    - Returns (feat, shifted_transcript, golden_transcript) on __getitem__.
    - Features have shape (num_feats, time).
    - Shifted transcripts prepend SOS token.
    - Golden transcripts append EOS token.
    - collate_fn pads features and transcripts to batch-uniform lengths.
    """
    def __init__(
        self,
        partition: str = 'train-clean-100',
        config: Optional[dict] = None,
        tokenizer: Optional[H3Tokenizer] = None,
        isTrainPartition: bool = True,
        global_stats: Optional[Tuple[Union[np.ndarray, jax.Array], Union[np.ndarray, jax.Array]]] = None,
        root: Optional[str] = None
    ):
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
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Directory paths
        self.partition_dir = os.path.join(self.root, partition)
        # TODO: Use root and partition to get the feature directory
        self.fbank_dir = os.path.join(self.partition_dir, 'fbank')
        
        # TODO: Get all feature files in the feature directory in sorted order
        if os.path.exists(self.fbank_dir):
            self.fbank_files = sorted([
                f for f in os.listdir(self.fbank_dir) if f.endswith('.npy')
            ])
        else:
            self.fbank_files = []

        # TODO: Take subset
        subset = self.config.get('subset', 1.0)
        total_files = len(self.fbank_files)
        if isinstance(subset, float):
            if not (0 < subset <= 1.0):
                raise ValueError("subset as float must be in (0, 1]")
            subset_size = max(1, int(total_files * subset))
        else:
            subset_size = int(subset)
        if subset_size < 1:
            raise ValueError("subset must be >= 1 when given as an integer")
        subset_size = min(subset_size, total_files)
        self.fbank_files = self.fbank_files[:subset_size]

        # TODO: Get the number of samples in the dataset
        self.length = len(self.fbank_files)

        # TODO: Use root and partition to get the text directory
        self.text_dir = os.path.join(self.partition_dir, 'text')
        if self.partition != 'test-clean' and os.path.exists(self.text_dir):
            # TODO: Get all text files in the text directory in sorted order
            self.text_files = sorted([
                f for f in os.listdir(self.text_dir) if f.endswith('.npy')
            ])
            # TODO: Take subset
            self.text_files = self.text_files[:subset_size]
        else:
            self.text_files = []

        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []
        self.feat_max_len = 0
        self.text_max_len = 0
        self.total_chars = 0
        self.total_tokens = 0

        num_feats = self.config.get('num_feats', 80)
        norm_type = self.config.get('norm', 'cepstral')

        # When norm is global_mvn and global_stats is None, update Welford
        # accumulators (count, mean, M2) in numpy. Otherwise store the provided stats.
        # Global MVN setup
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

        # Load samples into memory
        for i in range(self.length):
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            # TODO: Load features
            # Features are of shape (num_feats, time)
            feat = np.load(feat_path)
            # TODO: Truncate features to num_feats set by you in the config
            feat = feat[:num_feats, :]
            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update Welford statistics
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
                raw = np.load(text_path, allow_pickle=True)
                if isinstance(raw, np.ndarray) and raw.dtype.kind in ('U', 'S', 'O'):
                    transcript_str = "".join(raw.tolist())
                else:
                    transcript_str = str(raw.item() if hasattr(raw, 'ndim') and raw.ndim == 0 else raw)
                
                # TODO: Track character count (before tokenization)
                self.total_chars += len(transcript_str)
                # TODO: Use tokenizer to encode the transcript (see tokenizer.encode for details)
                tokens = self.tokenizer.encode(transcript_str)
                self.total_tokens += len(tokens)
                self.text_max_len = max(self.text_max_len, len(tokens) + 1)

                # TODO: Create shifted and golden versions by adding sos and eos tokens
                t_shifted = np.array([self.sos_token] + tokens, dtype=np.int64)
                t_golden = np.array(tokens + [self.eos_token], dtype=np.int64)
                self.transcripts_shifted.append(t_shifted)
                self.transcripts_golden.append(t_golden)

        # TODO: Compute final global_mean and global_std when using global_mvn
        if norm_type == 'global_mvn' and global_stats is None:
            variance = M2 / max(count - 1, 1)
            self.global_std = np.sqrt(variance + 1e-8).astype(np.float32)
            self.global_mean = mean.astype(np.float32)

        self.avg_chars_per_token = self.total_chars / max(self.total_tokens, 1)

    def get_avg_chars_per_token(self) -> float:
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        # TODO: Load features for this index as a float numpy array, shape (num_feats, time)
        feat = self.feats[idx].astype(np.float32)
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

        # TODO: Get transcripts for non-test partitions
        if self.partition == 'test-clean':
            shifted = None
            golden = None
        else:
            shifted = self.transcripts_shifted[idx]
            golden = self.transcripts_golden[idx]

        return feat, shifted, golden

    def collate_fn(self, batch: List[Tuple]):
        """
        Pads features and transcripts in the batch.
        
        Returns:
            Tuple: (batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lengths, transcript_lengths)
            where batch_feats_pad has shape (batch_size, max_feat_len, num_feats).
        """
        batch_size = len(batch)
        # TODO: Collect features from the batch. Each item is (num_feats, time);
        # the padded batch is (batch, max_time, num_feats), so transpose time to axis 1.
        feats = [item[0] for item in batch]  # each is (num_feats, time)
        num_feats = feats[0].shape[0]

        # TODO: Collect feature lengths from the batch
        feat_lens = np.array([f.shape[1] for f in feats], dtype=np.int64)
        max_feat_len = int(np.max(feat_lens))

        # TODO: Pad features to create a batch of fixed-length padded features
        batch_feats_pad = np.zeros((batch_size, max_feat_len, num_feats), dtype=np.float32)
        for i, f in enumerate(feats):
            batch_feats_pad[i, :f.shape[1], :] = f.T

        # TODO: Apply SpecAugment for training when config["specaug"] and isTrainPartition.
        # Features stay (batch, time, num_feats). Do not permute to a torch layout.
        if self.config.get("specaug", False) and self.isTrainPartition:
            conf = self.config.get("specaug_conf", {})
            # TODO: Apply frequency masking
            if conf.get("apply_freq_mask", False):
                num_mask = conf.get("num_freq_mask", 2)
                f_max = conf.get("freq_mask_width_range", 10)
                for _ in range(num_mask):
                    f = np.random.randint(0, f_max + 1)
                    f0 = np.random.randint(0, max(num_feats - f, 1))
                    batch_feats_pad[:, :, f0:f0 + f] = 0.0

            # TODO: Apply time masking
            if conf.get("apply_time_mask", False):
                num_mask = conf.get("num_time_mask", 2)
                t_max = conf.get("time_mask_width_range", 10)
                for _ in range(num_mask):
                    t = np.random.randint(0, t_max + 1)
                    t0 = np.random.randint(0, max(max_feat_len - t, 1))
                    batch_feats_pad[:, t0:t0 + t, :] = 0.0

        # TODO: Handle transcripts for non-test partitions
        if self.partition == 'test-clean':
            batch_shifted_pad = None
            batch_golden_pad = None
            transcript_lens = None
        else:
            shifteds = [item[1] for item in batch]
            goldens = [item[2] for item in batch]
            transcript_lens = np.array([len(s) for s in shifteds], dtype=np.int64)
            max_text_len = int(np.max(transcript_lens))

            batch_shifted_pad = np.full((batch_size, max_text_len), self.pad_token, dtype=np.int64)
            batch_golden_pad = np.full((batch_size, max_text_len), self.pad_token, dtype=np.int64)

            for i in range(batch_size):
                batch_shifted_pad[i, :len(shifteds[i])] = shifteds[i]
                batch_golden_pad[i, :len(goldens[i])] = goldens[i]

        # TODO: Return padded features, padded shifted, padded golden, feature lengths, and transcript lengths
        return batch_feats_pad, batch_shifted_pad, batch_golden_pad, feat_lens, transcript_lens


# Compatibility alias
ASRDataSource = ASRDataset
