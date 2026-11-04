# V2D Challenge 2026 — Track 2 (Robotic Grounding)

Public implementation repo for Kaggle team **Rohit Pardeshi** (NVIDIA).
Collaborator: **Nvidia-V2D-Challenge**.

Runtime base: [nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data) commit `0593d80de5d39ab6f56f73ed16ca464ba5053499` (FlashCHORD / G1+Dex3 / FlashSAC).

## Counted Kaggle submissions

Tier 1 mix CSV (`track2_tier1_hardmix.csv`): AUC 57625045, SP-SR 57625046, MP-SR 57625050, MPPE 57625052. Scores 0.434 / 0.533 / 0.300 / 13.04 cm.

Tier 2 16.8M CSV (`track2_tier2.csv`): AUC 57508092, SP-SR 57508094, MP-SR 57508095, MPPE 57508096. Scores 0.257 / 0.167 / 0.033 / 49.15 cm.

Tier 3 was not entered.

Object trajectories are policy rollouts (`evaluate_policy.py`, world_count=1, VOC=0), not reconstructions.

## Layout

- `scripts/` per-episode FlashSAC train/eval, mix, and pack helpers
- `configs/` ReconBody objective plus 350M / 16.8M / object / orientation curricula
- `receipts/` Kaggle upload receipts and the Tier 1 mix table
- `report/` 2-page technical report PDF
- `submissions/` counted Kaggle CSVs (same files as Kaggle IDs above)

Policy weights stay in the verification container, not this repo.
