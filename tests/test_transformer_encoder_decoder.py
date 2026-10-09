import os
import sys

# hw3lib lives in the directory that contains tests/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
from flax import nnx
from hw3lib.model import PadMask


def test_encoder_decoder_transformer(transformer):
    '''
    Test the EncoderDecoderTransformer implementation in JAX.
    '''
    # Structural Tests
    test_initialization(transformer)
    
    # Integration Tests
    test_encode_method(transformer)
    test_decode_method(transformer)
    test_forward_pass(transformer)
    test_encoder_decoder_integration(transformer)
    test_ctc_integration(transformer)
    test_forward_propagation_order(transformer)


def test_initialization(transformer):
    '''
    Test if the transformer is properly initialized.
    '''
    print("Testing initialization...")
    
    input_dim = 80
    time_reduction = 2
    reduction_method = 'both'
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.1
    max_len = 100
    num_classes = 10
    num_encoder_layers = 2
    num_decoder_layers = 2
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=time_reduction,
        reduction_method=reduction_method,
        num_encoder_layers=num_encoder_layers,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=num_decoder_layers,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes,
        rngs=nnx.Rngs(0)
    )

    expected_attributes = {
        "enc_layers", "dec_layers", "source_embedding", "target_embedding",
        "positional_encoding", "final_linear", "encoder_norm", "decoder_norm",
        "ctc_head"
    }
    assert expected_attributes.issubset(dir(model)), "Required components are missing"
    
    assert len(model.enc_layers) == num_encoder_layers, f"Expected {num_encoder_layers} encoder layers, got {len(model.enc_layers)}"
    assert len(model.dec_layers) == num_decoder_layers, f"Expected {num_decoder_layers} decoder layers, got {len(model.dec_layers)}"
    
    assert model.target_embedding.embedding_dim == d_model, "Target embedding dimension mismatch"
    assert model.target_embedding.num_embeddings == num_classes, "Target vocabulary size mismatch"
    
    print("Test Passed: All components initialized correctly")


def test_forward_pass(transformer):
    '''
    Test the forward pass of the transformer.
    '''
    print("Testing forward pass...")
    
    batch_size = 2
    input_seq_length = 100
    target_seq_length = 8
    input_dim = 80
    time_reduction = 2
    reduction_method = 'both'
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.0
    max_len = 100
    num_classes = 10
    num_encoder_layers = 2
    num_decoder_layers = 2

    model = transformer(
        input_dim=input_dim,
        time_reduction=time_reduction,
        reduction_method=reduction_method,
        num_encoder_layers=num_encoder_layers,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=num_decoder_layers,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes,
        rngs=nnx.Rngs(1)
    )
    
    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    source = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source_lengths = jax.random.randint(k2, (batch_size,), input_seq_length // 2, input_seq_length)
    targets = jax.random.randint(k3, (batch_size, target_seq_length), 0, num_classes)
    target_lengths = jax.random.randint(k4, (batch_size,), target_seq_length // 2, target_seq_length)
    
    output, attention_weights, _ = model(source, targets, source_lengths, target_lengths)
    
    assert output.shape == (batch_size, target_seq_length, num_classes), \
        f"Output shape mismatch: expected {(batch_size, target_seq_length, num_classes)}, got {output.shape}"
    
    for i in range(num_encoder_layers):
        enc_attn_key = f'layer{i+1}_enc_self'
        assert enc_attn_key in attention_weights, f"Missing encoder attention weights for {enc_attn_key}"
    
    for i in range(num_decoder_layers):
        dec_self_attn_key = f'layer{i+1}_dec_self'
        dec_cross_attn_key = f'layer{i+1}_dec_cross'
        assert dec_self_attn_key in attention_weights, f"Missing decoder self-attention weights for {dec_self_attn_key}"
        assert dec_cross_attn_key in attention_weights, f"Missing decoder cross-attention weights for {dec_cross_attn_key}"
    
    print("Test Passed: Forward pass works correctly")


def test_encoder_decoder_integration(transformer):
    '''
    Test the integration between encoder and decoder components.
    '''
    print("Testing encoder-decoder integration...")
    
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    d_model = 16
    num_encoder_heads = 4
    num_decoder_heads = 4
    d_ff_encoder = 32
    d_ff_decoder = 32
    dropout = 0.0
    max_len = 100
    num_classes = 10
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=num_encoder_heads,
        d_ff_encoder=d_ff_encoder,
        num_decoder_layers=2,
        num_decoder_heads=num_decoder_heads,
        d_ff_decoder=d_ff_decoder,
        d_model=d_model,
        dropout=dropout,
        max_len=max_len,
        num_classes=num_classes,
        rngs=nnx.Rngs(3)
    )
    
    key = jax.random.PRNGKey(4)
    k1, k2, k3 = jax.random.split(key, 3)
    source1 = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source2 = jax.random.normal(k2, (batch_size, input_seq_length, input_dim))
    source_lengths = jnp.ones(batch_size, dtype=jnp.int32) * input_seq_length
    
    targets = jax.random.randint(k3, (batch_size, target_seq_length), 0, num_classes)
    target_lengths = jnp.ones(batch_size, dtype=jnp.int32) * target_seq_length
    
    output1, attn1, ctc1 = model(source1, targets, source_lengths, target_lengths)
    output2, attn2, ctc2 = model(source2, targets, source_lengths, target_lengths)
    
    assert not jnp.allclose(output1, output2), "Different inputs should produce different outputs"
    assert not jnp.allclose(ctc1['log_probs'], ctc2['log_probs']), "Different inputs should produce different CTC outputs"
    
    for i in range(model.num_decoder_layers):
        cross_attn_key = f'layer{i+1}_dec_cross'
        assert not jnp.allclose(attn1[cross_attn_key], attn2[cross_attn_key]), \
            f"Cross-attention patterns should differ for layer {i+1}"
    
    print("Test Passed: Encoder-decoder integration works correctly")


def test_ctc_integration(transformer):
    '''
    Test the CTC head integration.
    '''
    print("Testing CTC integration...")
    
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=16,
        dropout=0.0,
        max_len=100,
        num_classes=10,
        rngs=nnx.Rngs(5)
    )
    
    key = jax.random.PRNGKey(6)
    k1, k2 = jax.random.split(key)
    source = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source_lengths = jnp.ones(batch_size, dtype=jnp.int32) * input_seq_length
    targets = jax.random.randint(k2, (batch_size, target_seq_length), 0, 10)
    target_lengths = jnp.ones(batch_size, dtype=jnp.int32) * target_seq_length
    
    _, _, ctc_input = model(source, targets, source_lengths, target_lengths)
    
    assert ctc_input['log_probs'].ndim == 3, "CTC output should be 3-dimensional"
    assert ctc_input['log_probs'].shape[2] == model.num_classes, "CTC output should have num_classes as last dimension"
    
    ctc_probs = jnp.exp(ctc_input['log_probs'])
    assert jnp.allclose(ctc_probs.sum(axis=-1), jnp.ones_like(ctc_probs.sum(axis=-1)), atol=1e-5), \
        "CTC probabilities should sum to 1"
    
    print("Test Passed: CTC integration works correctly")


def test_forward_propagation_order(transformer):
    '''
    Test that modules are called in the correct order during forward propagation.
    '''
    print("Testing forward propagation order...")
    
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=16,
        dropout=0.0,
        max_len=100,
        num_classes=10,
        encoder_pos_encoding='sinusoidal',
        decoder_pos_encoding='sinusoidal',
        rngs=nnx.Rngs(7)
    )
    
    key = jax.random.PRNGKey(8)
    k1, k2 = jax.random.split(key)
    source = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source_lengths = jnp.ones(batch_size, dtype=jnp.int32) * input_seq_length
    targets = jax.random.randint(k2, (batch_size, target_seq_length), 0, 10)
    target_lengths = jnp.ones(batch_size, dtype=jnp.int32) * target_seq_length
    
    execution_order = []
    class HookedWrapper:
        def __init__(self, inner, name):
            self._inner = inner
            self._name = name
        def __call__(self, *args, **kwargs):
            execution_order.append(self._name)
            return self._inner(*args, **kwargs)
        def __getattr__(self, attr):
            return getattr(self._inner, attr)

    model.source_embedding = HookedWrapper(model.source_embedding, 'source_embedding')
    model.positional_encoding = HookedWrapper(model.positional_encoding, 'pos_encoding')
    model.dropout = HookedWrapper(model.dropout, 'dropout')
    for i, layer in enumerate(model.enc_layers):
        model.enc_layers[i] = HookedWrapper(layer, f'encoder_layer_{i}')
    model.encoder_norm = HookedWrapper(model.encoder_norm, 'encoder_norm')
    model.ctc_head = HookedWrapper(model.ctc_head, 'ctc_head')
    model.target_embedding = HookedWrapper(model.target_embedding, 'target_embedding')
    for i, layer in enumerate(model.dec_layers):
        model.dec_layers[i] = HookedWrapper(layer, f'decoder_layer_{i}')
    model.decoder_norm = HookedWrapper(model.decoder_norm, 'decoder_norm')
    model.final_linear = HookedWrapper(model.final_linear, 'final_linear')
    
    model(source, targets, source_lengths, target_lengths)
    
    expected_order = [
        'source_embedding',
        'pos_encoding',
        'dropout',
        'encoder_layer_0',
        'encoder_layer_1',
        'encoder_norm',
        'ctc_head',
        'target_embedding',
        'pos_encoding',
        'dropout',
        'decoder_layer_0',
        'decoder_layer_1',
        'decoder_norm',
        'final_linear',
    ]
    
    assert execution_order == expected_order, \
        f"Incorrect execution order. Expected {expected_order}, got {execution_order}"
    
    print("Test Passed: Forward propagation order is correct")


def test_encode_method(transformer):
    '''
    Test the encode method of the transformer.
    '''
    print("Testing encode method...")
    
    batch_size = 2
    input_seq_length = 50
    input_dim = 80
    d_model = 16
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=d_model,
        dropout=0.0,
        max_len=100,
        num_classes=10,
        rngs=nnx.Rngs(9)
    )
    
    key = jax.random.PRNGKey(10)
    k1, k2 = jax.random.split(key)
    source = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source_lengths = jax.random.randint(k2, (batch_size,), input_seq_length // 2, input_seq_length)
    
    encoder_output, pad_mask_src, encoder_attention, ctc_input = model.encode(source, source_lengths)
    
    expected_enc_length = model.source_embedding.calculate_downsampled_length(jnp.ones(batch_size, dtype=jnp.int32) * input_seq_length)
    expected_enc_length = int(expected_enc_length[0])
    
    assert encoder_output.shape == (batch_size, expected_enc_length, d_model), \
        f"Encoder output shape mismatch: expected {(batch_size, expected_enc_length, d_model)}, got {encoder_output.shape}"
    assert jnp.array_equal(
        pad_mask_src, 
        PadMask(encoder_output, model.source_embedding.calculate_downsampled_length(source_lengths))
    ), "Pad mask source mismatch"

    assert ctc_input['log_probs'].shape == (expected_enc_length, batch_size, model.num_classes), \
        f"CTC output shape mismatch: expected {(expected_enc_length, batch_size, model.num_classes)}, got {ctc_input['log_probs'].shape}"
    
    for i in range(model.num_encoder_layers):
        attn_key = f'layer{i+1}_enc_self'
        assert attn_key in encoder_attention, f"Missing encoder attention weights for {attn_key}"
        assert encoder_attention[attn_key].shape == (batch_size, expected_enc_length, expected_enc_length), \
            f"Encoder attention weights shape mismatch for {attn_key}"
    
    ctc_probs = jnp.exp(ctc_input['log_probs'])
    assert jnp.allclose(ctc_probs.sum(axis=-1), jnp.ones_like(ctc_probs.sum(axis=-1)), atol=1e-5), \
        "CTC probabilities should sum to 1"
    
    print("Test Passed: Encode method works correctly")


def test_decode_method(transformer):
    '''
    Test the decode method of the transformer.
    '''
    print("Testing decode method...")
    
    batch_size = 2
    input_seq_length = 50
    target_seq_length = 8
    input_dim = 80
    d_model = 16
    
    model = transformer(
        input_dim=input_dim,
        time_reduction=2,
        reduction_method='both',
        num_encoder_layers=2,
        num_encoder_heads=4,
        d_ff_encoder=32,
        num_decoder_layers=2,
        num_decoder_heads=4,
        d_ff_decoder=32,
        d_model=d_model,
        dropout=0.0,
        max_len=100,
        num_classes=10,
        rngs=nnx.Rngs(11)
    )
    
    key = jax.random.PRNGKey(12)
    k1, k2, k3 = jax.random.split(key, 3)
    source = jax.random.normal(k1, (batch_size, input_seq_length, input_dim))
    source_lengths = jax.random.randint(k2, (batch_size,), input_seq_length // 2, input_seq_length)
    encoder_output, pad_mask_src, _, _ = model.encode(source, source_lengths)
    
    targets = jax.random.randint(k3, (batch_size, target_seq_length), 0, 10)
    target_lengths = jnp.ones(batch_size, dtype=jnp.int32) * target_seq_length
    
    decoder_output, decoder_attention = model.decode(targets, encoder_output, target_lengths, pad_mask_src)
    
    assert decoder_output.shape == (batch_size, target_seq_length, model.num_classes), \
        f"Decoder output shape mismatch: expected {(batch_size, target_seq_length, model.num_classes)}, got {decoder_output.shape}"
    
    expected_enc_length = model.source_embedding.calculate_downsampled_length(jnp.ones(batch_size, dtype=jnp.int32) * input_seq_length)
    expected_enc_length = int(expected_enc_length[0])
    for i in range(model.num_decoder_layers):
        self_attn_key = f'layer{i+1}_dec_self'
        assert self_attn_key in decoder_attention, f"Missing decoder self-attention weights for {self_attn_key}"
        assert decoder_attention[self_attn_key].shape == (batch_size, target_seq_length, target_seq_length), \
            f"Decoder self-attention weights shape mismatch for {self_attn_key}"
        
        cross_attn_key = f'layer{i+1}_dec_cross'
        assert cross_attn_key in decoder_attention, f"Missing decoder cross-attention weights for {cross_attn_key}"
        assert decoder_attention[cross_attn_key].shape == (batch_size, target_seq_length, expected_enc_length), \
            f"Decoder cross-attention weights shape mismatch for {cross_attn_key}"
    
    # Test causal masking in self-attention
    for i in range(model.num_decoder_layers):
        self_attn_key = f'layer{i+1}_dec_self'
        attn_weights = decoder_attention[self_attn_key]
        
        for t in range(target_seq_length):
            for future_t in range(t + 1, target_seq_length):
                assert jnp.all(attn_weights[:, t, future_t] == 0.0), \
                    f"Position {t} should not attend to future position {future_t}"
    
    print("Test Passed: Decode method works correctly")


def main():
    '''
    Main function to run the tests
    '''
    from hw3lib.model import EncoderDecoderTransformer
    from tests.testing_framework import TestingFramework

    framework = TestingFramework(
        test_categories={
            'EncoderDecoderTransformer': [
                {
                    'func': lambda: test_encoder_decoder_transformer(EncoderDecoderTransformer),
                    'description': 'Test the encoder-decoder transformer in JAX'
                }
            ]
        }
    )
    framework.run_tests()
    framework.summarize_results()


if __name__ == '__main__':
    main()
