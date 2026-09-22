"""Show the original kernel's signed address overflow without reading OOB."""
import mlx.core as mx

probe = mx.fast.metal_kernel(
    name="sparse_address_overflow_probe",
    input_names=["lengths"], output_names=["old", "wide"],
    source=r"""
        uint i = thread_position_in_grid.x;
        int key_length = lengths[i];
        int batch_idx = 0;
        int kv_head_idx = 63;
        int key_pos = key_length - 1;
        // Exactly the original sparse kernel's K/V element address expression.
        old[i] = (((batch_idx * 64 + kv_head_idx) * key_length + key_pos) * 256);
        wide[i] = (((size_t(batch_idx) * 64 + kv_head_idx) * key_length + key_pos) * 256);
    """,
)
lengths = [131072, 131073, 189055, 192372, 262145, 1048576]
old, wide = probe(inputs=[mx.array(lengths)], grid=(len(lengths), 1, 1), threadgroup=(len(lengths), 1, 1), output_shapes=[(len(lengths),)] * 2, output_dtypes=[mx.int64, mx.uint64])
for n, a, b in zip(lengths, old.tolist(), wide.tolist()):
    expected = (64 * n - 1) * 256
    assert b == expected
    print(f"tokens={n} original={a} wide={b} expected={expected}", flush=True)
