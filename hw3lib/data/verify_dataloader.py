"""
Dataloader Verification Utility for ASR Datasets.
"""

def verify_dataloader(dataloader):
    """
    Computes the maximum sequence lengths of features and text in the dataloader.  
    Prints basic statistics about the dataloader.
    """
    def print_shapes(feats, transcripts_shifted=None, transcripts_golden=None, feat_lengths=None, transcript_lengths=None):
        if feats is not None:
            print(f"{'Feature Shape':<25}: {list(feats.shape)}")
        if transcripts_shifted is not None:
            print(f"{'Shifted Transcript Shape':<25}: {list(transcripts_shifted.shape)}")
        if transcripts_golden is not None:
            print(f"{'Golden Transcript Shape':<25}: {list(transcripts_golden.shape)}")
        if feat_lengths is not None:
            print(f"{'Feature Lengths Shape':<25}: {list(feat_lengths.shape)}")
        if transcript_lengths is not None:
            print(f"{'Transcript Lengths Shape':<25}: {list(transcript_lengths.shape)}")

    print("=" * 50)
    print(f"{'Dataloader Verification':^50}")
    print("=" * 50)
    partition = getattr(dataloader.dataset, 'partition', 'Unknown')
    print(f"{'Dataloader Partition':<25}: {partition}")
    print("-" * 50)
    num_batches = len(dataloader) if hasattr(dataloader, '__len__') else 'N/A'
    batch_size = getattr(dataloader, 'batch_size', 'N/A')
    print(f"{'Number of Batches':<25}: {num_batches}")
    print(f"{'Batch Size':<25}: {batch_size}")
    print("-" * 50)
    print(f"{'Checking shapes of the data...':<50}\n")

    max_feat_len = None
    max_transcript_len = None

    for i, batch in enumerate(dataloader):
        if i > 0:
            break

        if hasattr(dataloader.dataset, 'feat_max_len'):
            feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths = batch
            print_shapes(feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths)
            max_feat_len = dataloader.dataset.feat_max_len
            max_transcript_len = dataloader.dataset.text_max_len
        elif hasattr(dataloader.dataset, 'text_max_len'):
            transcripts_shifted, transcripts_golden, lengths = batch
            print_shapes(None, transcripts_shifted, transcripts_golden, None, lengths)
            max_transcript_len = dataloader.dataset.text_max_len
        else:
            feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths = batch
            print_shapes(feats, transcripts_shifted, transcripts_golden, feat_lengths, transcript_lengths)

    print("-" * 50)
    if max_feat_len is not None:
        print(f"{'Max Feature Length':<25}: {max_feat_len}")
    if max_transcript_len is not None:
        print(f"{'Max Transcript Length':<25}: {max_transcript_len}")
    if hasattr(dataloader.dataset, 'get_avg_chars_per_token'):
        print(f"{'Avg. Chars per Token':<25}: {dataloader.dataset.get_avg_chars_per_token():.2f}")
    print("=" * 50)
