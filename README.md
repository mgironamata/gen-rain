# gen-rain
Deep generative models for rainfall generation.

The `GaussianRainfieldGenerator` can now return either binary occurrence
masks or rainfall amounts depending on the `occurrence_only` flag in
`sample_precip` and `make_dataset`.

Training and sampling pipelines in `models.py` accept the same option.
When ``occurrence_only=True`` only the discrete wet/dry diffusion is used,
allowing lightweight models that predict rainfall occurrence without
intensity values.

Rainfall intensities may optionally be drawn from a correlated log-normal
field by setting ``correlated_intensity=True`` when constructing the
generator.  This uses a second Gaussian process to sample log-rainfall,
providing spatially coherent amounts on wet pixels.
