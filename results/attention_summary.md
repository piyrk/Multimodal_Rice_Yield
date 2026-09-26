# Attention Weight Summary

These are descriptive learned attention weights from the full Satellite + Weather + Soil + Attention model. They are not causal importance measures.

Each seed row summarizes the 41 held-out test samples. The `all_seeds` rows summarize 123 attention vectors across the three repeatability runs.

| Scope | Modality | N | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| seed_42 | satellite | 41 | 0.393527 | 0.002546 | 0.383237 | 0.396503 |
| seed_42 | weather | 41 | 0.350018 | 0.002422 | 0.344758 | 0.356662 |
| seed_42 | soil | 41 | 0.256455 | 0.001491 | 0.253951 | 0.260101 |
| seed_123 | satellite | 41 | 0.411354 | 0.032893 | 0.370755 | 0.502147 |
| seed_123 | weather | 41 | 0.396176 | 0.024267 | 0.333959 | 0.437443 |
| seed_123 | soil | 41 | 0.192469 | 0.011202 | 0.163580 | 0.206722 |
| seed_2026 | satellite | 41 | 0.359451 | 0.106611 | 0.200708 | 0.572651 |
| seed_2026 | weather | 41 | 0.316217 | 0.072412 | 0.198827 | 0.522456 |
| seed_2026 | soil | 41 | 0.324332 | 0.054747 | 0.203307 | 0.400505 |
| all_seeds | satellite | 123 | 0.388111 | 0.067935 | 0.200708 | 0.572651 |
| all_seeds | weather | 123 | 0.354137 | 0.054956 | 0.198827 | 0.522456 |
| all_seeds | soil | 123 | 0.257752 | 0.062773 | 0.163580 | 0.400505 |

Attention weights are normalized across the three modalities for each sample by the model's softmax attention layer.