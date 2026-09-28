# Required teacher checkpoints

Weights are intentionally excluded from the repository. Place these files here:

```text
Mask2Former_hornet_3x_576d0b.pth
clip_b16_grit20m_fultune_2xe.pth
```

The validated files used for the reported experiments had these SHA-256 hashes:

```text
08c637c3e9f26c4ee3f264ce867f8aa953c57273bb2c86180deb441311b634c5  Mask2Former_hornet_3x_576d0b.pth
6ff9e3c5735aa34bcbd386ba4d1b99e5efc5deffab077d3d2f2911d1f1c43a57  clip_b16_grit20m_fultune_2xe.pth
```

Obtain the weights from the CropFormer/Entity and Alpha-CLIP upstream projects
and follow their licenses. Do not substitute a different CLIP checkpoint when
comparing reported semantic metrics.
