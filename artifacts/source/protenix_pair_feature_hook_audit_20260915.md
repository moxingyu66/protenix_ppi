# Protenix pair-feature hook audit (planning evidence)

This is a source-level audit of commit `4c355be4553512f72453ecbfb65e69f4c35d1413`.
It is not a G1 run and is not the final B2 feature lock.

The most defensible primary B2 hook candidate is the `z` object returned by
`Protenix.get_pairformer_output(...)`. In the inspected source it is the final
recycled Pairformer pair representation, returned as the third element of
`(s_inputs, s, z)`, with shape `[..., N_token, N_token, c_z]`. The inference loop
passes this tensor to the diffusion and confidence paths, so it is a genuine
pair representation rather than only `iptm` or `ranking_score`.

The proposed label-blind pooling is to retain cross-protein token pairs, average
each pair direction with its transpose, and concatenate per-channel mean and
standard deviation. Chain-order swap invariance must be measured on the actual
G1 input and registered with a tolerance before any B2 extraction.

G1 must re-check the exact module path, configured `c_z`, recycle timing, dtype,
MSA/template policy, and tensor shape on the executed installation. Only then
may `configs/b2_feature_extraction_lock.template.json` be filled and feature
extraction begin.
