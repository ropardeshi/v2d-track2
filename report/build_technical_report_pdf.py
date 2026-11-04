#!/usr/bin/env python3
"""Two-page V2D Track 2 technical report PDF."""
from pathlib import Path

from fpdf import FPDF

OUT = Path("/home/horde/isaac-automation/v2d_track2/V2D_Track2_Technical_Report.pdf")


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_I = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"


class Report(FPDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_font("Serif", "", FONT)
        self.add_font("Serif", "B", FONT_B)
        self.add_font("Serif", "I", FONT_I)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Serif", "I", 8)
        self.set_text_color(80)
        self.cell(0, 6, "Pardeshi - Residual FlashSAC Grounding for V2D Track 2", align="C")
        self.ln(8)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Serif", "I", 8)
        self.cell(0, 8, str(self.page_no()), align="C")

    def heading(self, text):
        self.set_x(self.l_margin)
        self.set_font("Serif", "B", 11)
        self.ln(1.5)
        self.multi_cell(0, 5.2, text)
        self.ln(0.6)

    def p(self, text):
        self.set_x(self.l_margin)
        self.set_font("Serif", "", 10)
        self.multi_cell(0, 4.7, text)
        self.ln(0.8)

    def bi(self, label, text):
        self.set_font("Serif", "B", 10)
        self.write(4.7, label)
        self.set_font("Serif", "", 10)
        self.multi_cell(0, 4.7, text)


def main() -> None:
    pdf = Report(format="letter", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Serif", "B", 14)
    pdf.multi_cell(
        0,
        6,
        "Residual FlashSAC Grounding of 4D Human-Object Motion onto G1+Dex3",
        align="C",
    )
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Serif", "", 11)
    pdf.multi_cell(0, 5, "V2D Challenge 2026 - Track 2 Technical Report", align="C")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Serif", "I", 10)
    pdf.multi_cell(
        0,
        5,
        "Rohit Pardeshi  |  NVIDIA SimuNex  |  Kaggle team: Rohit Pardeshi  |  ropardeshi@nvidia.com",
        align="C",
    )
    pdf.ln(2)

    pdf.heading("1. Introduction")
    pdf.p(
        "Track 2 of the NVIDIA Isaac Video-to-Data (V2D) Challenge asks whether a fixed "
        "humanoid embodiment can replay 4D human-object interaction (HOI) as a learned "
        "policy. Embodiment and simulator are given: Unitree G1 with Dex3 hands in "
        "FlashCHORD (Newton / Warp / JAX). The scored output is the object pose the "
        "policy actually produces, not a reconstructed or copied reference. All three "
        "input tiers are measured against the same clean multi-view object trajectory "
        "(30 episodes, 22,989 frames at 30 Hz, 29 unique objects)."
    )
    pdf.p(
        "We entered Tier 1 (clean SOMA-X / multi-view captions) and Tier 2 (the official "
        "synthetically corrupted reference). We did not enter Tier 3. Trajectories are "
        "policy rollouts from evaluate_policy.py with world_count=1 and VOC scale 0. We "
        "did not use Tier 1 captions on Tier 2, and we did not submit reconstructions as "
        "rollouts."
    )

    pdf.heading("2. Approach")
    pdf.p(
        "Pipeline. Released LeRobot episodes are converted to SOMA-X folders, retargeted "
        "to G1+Dex3 with the official soma_to_g1 path, and trained with FlashSAC on the "
        "ReconBody objective. Support geometry (USD cylinders/cubes) is enabled only when "
        "the retargeted USDA actually contains it; empty support files are left off. One "
        "policy is trained per episode. Evaluation writes an object-pose parquet per "
        "episode; the official packer converts those poses into the released world frame "
        "with one rigid transform and emits a single CSV for that tier’s four Kaggle boards."
    )
    pdf.p(
        "Learning. The legal FlashSAC recipe uses 2048 parallel worlds. We ran three "
        "budgets on the same retargeted motion: a short 0.5M-step probe, a 16.8M-step "
        "compressed-VOC schedule, and the official 350M-step VOC curriculum "
        "(recon_body_flash_sac). Residual FlashSAC acts on SONIC joints with scale 0.15. "
        "A later hard-episode curriculum (recon_body_flash_sac_object) raises object-"
        "position and palm-orientation weights for clips that still drop or twist. An "
        "object-orientation geodesic term was added to ReconBody for a final near-miss "
        "pass; those policies were not mixed into the counted Kaggle file."
    )
    pdf.p(
        "Selection. For each of the 30 episodes we keep the rollout with the best "
        "CHORD gates (SP-SR then MP-SR) and lowest MPPE among the short, 16.8M, 350M, "
        "and hard runs. The counted Tier 1 file is this mix (22×350M, 4×16.8M, 2×short, "
        "2×hard). Tier 2 uses one 16.8M FlashSAC policy per episode trained on the noisy "
        "reference only."
    )

    pdf.heading("3. Experiments and Results")
    pdf.p(
        "Training ran on RTX PRO 6000 GPUs, about two hours per 350M episode. Eval is "
        "deterministic (actor mean). Local packer scores match Kaggle. Counted submissions:"
    )
    pdf.set_font("Serif", "B", 9.5)
    widths = [52, 28, 28, 28, 32]
    headers = ["Entry", "AUC up", "SP-SR up", "MP-SR up", "MPPE cm down"]
    for j, header in enumerate(headers):
        pdf.cell(
            widths[j],
            5.5,
            header,
            border=1,
            align="L" if j == 0 else "C",
            new_x="LMARGIN" if j == 4 else "RIGHT",
            new_y="NEXT" if j == 4 else "TOP",
        )
    pdf.set_font("Serif", "", 9.5)
    rows = [
        ("Tier 1 350M uniform", "0.409", "0.467", "0.267", "17.98"),
        ("Tier 1 mix (final)", "0.434", "0.533", "0.300", "13.04"),
        ("Tier 2 16.8M (final)", "0.257", "0.167", "0.033", "49.15"),
    ]
    for i, row in enumerate(rows):
        fill = i == 1
        if fill:
            pdf.set_fill_color(230, 230, 230)
        for j, cell in enumerate(row):
            w = [52, 28, 28, 28, 32][j] if j == 0 else [52, 28, 28, 28, 32][j]
            pdf.cell(
                widths[j],
                5.5,
                cell,
                border=1,
                align="L" if j == 0 else "C",
                fill=fill,
                new_x="LMARGIN" if j == 4 else "RIGHT",
                new_y="NEXT" if j == 4 else "TOP",
            )
    pdf.ln(2)
    pdf.p(
        "On the counted Tier 1 mix, 16/30 episodes pass SP-SR (≤10 cm and ≤0.5 rad) and "
        "9/30 pass MP-SR (≤3 cm and ≤30°). The mix recovers two failures that uniform "
        "350M missed: tall-bar-stool pull (episode 21) reaches SP after the object/palm "
        "retrain (5.7 cm, 13.5°), and squeegee (episode 28) drops from 41 cm to 7.3 cm "
        "MPPE. Four 16.8M rollouts beat their 350M counterparts (episodes 0, 1, 7, 22), "
        "so longer training is not uniformly better."
    )
    pdf.p(
        "Residual errors are concentrated. Baseball bat and cyan water bottle remain "
        "near 1 m MPPE across every budget — thin, rotating objects that the G1 grasp "
        "never stabilizes. Vacuum, easter basket, and snack box sit at 20–39 cm. Several "
        "near-misses are millimetres or tenths of a degree from a gate (trash-can drag "
        "3.12 cm vs the 3 cm MP line; book 28.78° vs the 28.6° SP line). Those clips "
        "are orientation- or position-gated, not collapsed."
    )
    pdf.p(
        "Tier 2, trained only on the noisy reference, keeps the same embodiment and "
        "packer but loses most of the success-rate mass (SP 5/30, MP 1/30). That gap is "
        "the Track 2 measurement: the same FlashSAC recipe is much less tolerant of "
        "upstream pose noise than of clean SOMA-X. We did not recover a better Tier 2 "
        "trajectory from video, and we did not peek at Tier 1 captions."
    )

    pdf.heading("4. Conclusion")
    pdf.p(
        "A legal G1 FlashSAC stack, per-episode policies, and a best-of mix across "
        "training budgets produce a valid Track 2 policy-rollout submission. Scaling to "
        "the official 350M VOC schedule is the main lift; mixing shorter runs and a "
        "targeted object/palm retrain recovers additional SP/MP mass and cuts mean MPPE "
        "from 18 cm to 13 cm. The remaining failures are a small set of unstable objects, "
        "not a global retargeting bug. Code used at evaluation time is "
        "nvidia-isaac/video_to_data commit 0593d80, plus the per-episode FlashSAC "
        "launch and mix scripts in our training repository."
    )

    pdf.heading("Industry innovation (judging areas)")
    pdf.bi(
        "Economic viability. ",
        "One FlashCHORD recipe, no Isaac Sim/Lab in the scored loop, and about two GPU-"
        "hours per 350M episode. Best-of mixing reuses finished rollouts instead of "
        "retraining the full set.",
    )
    pdf.bi(
        "Data flywheel. ",
        "Each new budget (0.5M, 16.8M, 350M, hard) is another candidate per episode. "
        "The mix table is the flywheel: later runs replace a clip only when they beat "
        "the stored gates and MPPE.",
    )
    pdf.bi(
        "Complexity reduction. ",
        "Fixed G1+Dex3, official retargeter, official evaluator and packer. No custom "
        "simulator, no reconstructed meshes in the scored file, one CSV per tier.",
    )
    pdf.bi(
        "Innovation. ",
        "Per-episode residual FlashSAC with a CHORD-aware mix, plus a stronger object/"
        "palm curriculum on the clips that still fail after 350M, rather than a single "
        "global policy or a copied reference trajectory.",
    )

    pdf.ln(2)
    pdf.set_font("Serif", "I", 8)
    pdf.multi_cell(
        0,
        4,
        "Kaggle (final): Track 2 Tier 1 57625045 / 57625046 / 57625050 / 57625052 "
        "(AUC / SP-SR / MP-SR / MPPE). Track 2 Tier 2 57508092 / 57508094 / 57508095 / "
        "57508096. Runtime commit: https://github.com/nvidia-isaac/video_to_data/commit/"
        "0593d80de5d39ab6f56f73ed16ca464ba5053499. FlashCHORD image flash_chord:latest.",
    )

    pdf.output(OUT)
    print(f"wrote {OUT} pages={pdf.page_no()}")


if __name__ == "__main__":
    main()
