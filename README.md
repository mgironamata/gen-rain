# gen-rain
Deep generative models for rainfall generation.

The `GaussianRainfieldGenerator` can now return either binary occurrence
masks or rainfall amounts depending on the `occurrence_only` flag in
`sample_precip` and `make_dataset`.

Rainfall intensities may optionally be drawn from a correlated log-normal
field by setting ``correlated_intensity=True`` when constructing the
generator.  This uses a second Gaussian process to sample log-rainfall,
providing spatially coherent amounts on wet pixels.
