# Collective State-Space Possession Inference Under Severe Ball Occlusion

**More Than a Hack 2027 — Initial Challenge Technical Document**

---

## 1. The Approach

### Problem Formulation
In the Initial Challenge (`FC Barcelona vs. Athletic Club`, 199,319 frames at 25 Hz), the central ranking metric is **`spellF1`** — temporal Intersection-over-Union (IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$) over maximal contiguous runs of live possession (`H` or `A`), where dead-ball stretches (`D`) act as boundary separators between possessions.

Optimizing for `spellF1` differs fundamentally from optimizing per-frame accuracy (`macroF1`):
1. **Zero tolerance for label flicker:** A single spurious frame of `A` or `D` in the middle of an `H` possession splits one true possession into two false positives and destroys temporal IoU.
2. **Stoppage boundaries (`D`) are as critical as team switches:** Missing a stoppage merges two consecutive same-team possessions into one over-extended spell, costing both recall and precision.
3. **40.7% missing ball & anonymous tracklets:** The ball is absent on 40.7% of graded frames (disproportionately during duels, aerial balls, and stoppages), and players are represented by 46 anonymous, identity-swapping track slots (~20.5 active per frame).

### Three-Stage Architecture
We built a deterministic, zero-leakage pipeline (`possession/`) trained and cross-validated exclusively on **open Metrica Sports matches (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`)** degraded through a calibrated Automatic Tracking Data (ATD) noise simulator:

1. **Collective Spatio-Temporal Feature Extraction (`possession/features.py`):**
   Because track slots swap identities, all features are permutation-invariant across slots within each team:
   - **Short-gap ball interpolation:** Linear interpolation across short ball occlusions ($\le 1.0\text{ s}$, 25 frames) within a half, plus explicit indicator flags (`seen`, `has_ball`, `since_seen`).
   - **Ball-proximity & local pressure geometry:** Distance of the nearest Home (`dh`) and Away (`da`) player to the ball, hyperbolic-tangent distance advantage $\tanh((d_A - d_H)/3\text{ m})$, and player counts within a $5\text{ m}$ pressure radius (`n_h5`, `n_a5`).
   - **Out-of-bounds & kinematics:** Signed margin outside the $105.3 \times 68.0\text{ m}$ pitch boundary (`out_margin`), smoothed ball speed (`bspeed`), and collective player speed across all active slots (`mean_speed`, 1 s and 4 s rolling averages), with slot jumps $> 12\text{ m/s}$ masked out as identity switches.
   - **Collective Team Shape (crucial during the 41% ball-missing stretches):** Attacking direction is inferred automatically per half from the median separation of the two team centroids (requiring zero labels). In the attacking frame, we compute each team's longitudinal centroid (`home_x`, `away_x`), block width (`home_wid`, `away_wid`), and block depth (`home_len`, `away_len`) smoothed over 3 s windows. A team in possession expands in width and pushes its centroid higher up the pitch, allowing the model to track possession state even when the ball is invisible for 10+ seconds.

2. **Prior-Corrected Multinomial Emission Model (`possession/methods.py` — `LogReg`):**
   A 29-feature $L_2$-regularized multinomial logistic regression classifier predicts per-frame state posteriors $P(y_t \in \{\text{H}, \text{A}, \text{D}\} \mid x_t)$. We convert posteriors to scaled log-likelihood ratios by subtracting the log class prior $\log P(y)$, preventing majority-class bias during long occlusions.

3. **Global Viterbi State-Space Decoding (`possession/methods.py` — `viterbi`):**
   Instead of independent per-frame decisions or naive sliding-window majority votes (`baseline_smooth50`), we decode each half globally via dynamic programming (Viterbi) over the three states $\{\text{H}, \text{A}, \text{D}\}$ with temperature scaling ($\beta = 1.0$) and asymmetric transition penalties:
   - Direct possession turnover cost $C(\text{H} \leftrightarrow \text{A}) = 8.0$
   - Stoppage entry cost $C(\text{H/A} \to \text{D}) = 32.0$
   - Restart cost $C(\text{D} \to \text{H/A}) = 32.0$
   - Frames outside the graded half intervals (`half_video_blocks`) are assigned `X`.

---

## 2. Assets Used (and What We Tested & Dropped)

| Asset | Status | How it was used / why it was dropped |
|---|---|---|
| **Tactical ATD (`BAR-ATH_tactical_atd.xml/.txt`)** | **Used (Primary)** | Provides all 199,319 frames at 25 Hz, 46 player track slots, and raw ball coordinates. |
| **`games.py` metadata (`half_video_blocks`, pitch $105.3 \times 68.0\text{ m}$)** | **Used** | Defines exact graded frame intervals (`[11185, 90651]` and `[110808, 190197]`) so non-graded frames are cleanly marked `X`, and supplies exact pitch dimensions for out-of-bounds detection. |
| **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`)** | **Built, excluded from the default** | We built an unsupervised polarity calibrator (`possession/tagging.py`) that aligns phase codes with tracking-only possession estimates, and per-frame features for set-piece windows and phase team. They are excluded by default for reasons that do not depend on the provided game: (1) phase intervals are phases, not possessions — a phase can span one possession, stop short of one, or cover two (`DATA.md`); (2) the meaning of a phase's `Team` tag is not documented for every code, so its polarity has to be guessed; (3) the open training games have no Smart Tagging, so tag features can only be trained on our own simulation of tags (`tagging.simulate`), which encodes exactly those guesses. *Disclosure:* the question came up when the pipeline check on the public 5-minute window was run with and without tags (`0.3314` vs `0.5200` `spellF1`). That window is not a tuning set, and 17 possessions are far too few for a difference to be evidence, so the decision rests on (1)–(3), not on that number. Tags remain available via `--use-tags`. |
| **Broadcast ATD & Video** | **Not used** | The tactical ATD covers 100% of frames and is the reference timeline, whereas the broadcast feed is not synchronised and has usable homography on 65.8% of its frames (`DATA.md`). We did not test it. |
| **Tactical Homography** | **Not used** | The tactical ATD is already in pitch coordinates. The homography would only be needed for a video-based ball detector (Section 5). |

---

## 3. Training Data & Label-Free ATD Noise Calibration

In strict compliance with `RULES.md`:
- **No labels from the provided game** were used for training, hyperparameter selection, or threshold tuning. The public 5-minute window was used only as a pipeline check (frames aligned, file well-formed).
- **The only information taken from the provided game is label-free:** aggregate kinematic distributions of its tracking data, used to calibrate the noise simulator (below). `RULES.md` permits learning without human labels.
- **Training corpus:** The three public Metrica Sports open games (`github.com/metrica-sports/sample-data`: `Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`), totalling ~420,000 frames across three full matches. Ground-truth `H`/`A`/`D`/`X` labels are derived mechanically from the open event streams (`possession/devgames.py`) following the exact rules in `CHALLENGE.md`.
- **Realistic ATD Noise Simulation (`possession/noise.py`):** Clean open tracking data differs sharply from raw computer-vision ATD. Using a label-free diagnostic (`possession/diagnose.py`) that compares only aggregate kinematic distributions (gaps per minute, gap duration quantiles, and player speed quantiles), we discovered two critical physical properties of real ATD:
  1. **Ball dropouts occur in long contiguous bursts** (1.4 gaps/min, median gap 10.7 s, 90th percentile 33.8 s) rather than sub-second flickers.
  2. **Player trajectories are temporally smooth** (median player speed $1.54\text{ m/s}$, 10th percentile $0.71\text{ m/s}$ during stoppages). Over-jittering synthetic player positions inflates resting player speeds above $1.6\text{ m/s}$ and blinds the stoppage (`D`) detector.
  Calibrating `NoiseConfig` (`gap_median_frames=250`, `player_jitter_m=0.04`, across three presets `mid`, `alive_hard`, `dead_dark`) matched these label-free distributions (`python -m possession.diagnose` prints both side by side).

### Leave-One-Game-Out Benchmark (Open Metrica Games 1–3)

`python -m possession.bench --quick`: each open game is held out once; the model is trained on the other two, degraded with the calibrated `mid` noise preset, and the Viterbi costs are grid-searched on the training games only (grid `{8, 32}` for each cost). All three folds select the same costs (`switch=8`, `stop=32`, `start=32`), which are the ones used for the submission.

| Method | Mean `spellF1` (min) | Mean `macroF1` | Predicted / true possessions | Description |
|---|---|---|---|---|
| `v0` (Baseline reproduction) | `0.293` (`0.287`) | `0.667` | 1,530 / 1,064 | Nearest player + $\pm 2\text{ s}$ majority vote (`baseline_smooth50`) |
| `v1` (Rule-based memory) | `0.378` (`0.361`) | `0.770` | 975 / 1,064 | Ball-control radius ($1.5\text{ m}$, $\ge 5$ frames) + collective slowdown stoppage rule |
| **`v2` (Collective LogReg + Viterbi)** | **`0.493`** (`0.485`) | **`0.857`** | 824 / 1,064 | **29 collective features + prior-corrected multinomial emissions + Viterbi decoding** |

`v2` predicts about 23% fewer possessions than there are: it merges possessions across missed turnovers and short stoppages (Section 5).

---

## 4. Possession Rules Implemented

1. **Controlled Possession vs. Deflections / Rebounds:**
   A single proximity frame or high-speed ball deflection ($v_\text{ball} > 8\text{ m/s}$) does not trigger a possession switch. In `v1`, a player must remain within $1.5\text{ m}$ of a controlled ball ($< 8\text{ m/s}$) for at least 5 consecutive frames ($0.2\text{ s}$); in `v2`, the Viterbi transition penalty ($C(\text{H} \leftrightarrow \text{A}) = 8.0$) requires sustained emission evidence (~0.4–0.8 s of proximity, support within $5\text{ m}$, and team shape shift) before committing to a turnover.
2. **Loose & Contested Balls:**
   While the ball travels between passes or is occluded during an aerial duel, possession stays with the last controlling team via Viterbi state persistence and collective team-shape features (`wid_diff_3s`, `x_diff_3s`).
3. **Dead Ball (`D`) Entry and Restart:**
   A stoppage begins when the ball leaves the $105.3 \times 68.0\text{ m}$ pitch (`out_margin > 0.5 m`) or when collective player speed across the pitch drops (`mean_speed_4s < 1.2–1.6 m/s`) while the ball is uncontrolled or unseen. Notice that in the 2nd half, grading starts at frame `110,808` while kickoff occurs at `113,742`; collective speed and missing-ball features naturally classify this pre-kickoff interval as `D`.

---

## 5. Limitations & Future Work

- **Too few possessions:** On the open games `v2` finds about 77% of the true number of possessions (824 of 1,064), so some turnovers and short stoppages are missed and two possessions merge into one.
- **Long-occlusion turnovers:** When a turnover occurs inside a 15+ second ball-tracking dropout, the model relies on collective team shape (`x_diff_3s`, `wid_diff_3s`), which lags the exact touch by ~1–2 seconds and lowers IoU at the strictest `0.7` threshold.
- **Set-piece execution frame precision:** Players often stand still for several seconds before a free kick or corner is struck; distinguishing the walk-up from the exact restart frame without ball visibility can shift the `D -> H/A` boundary by 10–20 frames.
- **With more time:** We would run a lightweight pretrained ball detector locally on the 1080p tactical video specifically during the 40.7% ball-missing intervals to recover exact restart and interception timestamps.

---

## 6. Exact Reproduction

From a clean clone of the repository with the official data bundle unpacked in `data/`:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\python -m pip install -r requirements-team.txt
.venv\Scripts\python -m possession.run --team <team_name>
.venv\Scripts\python evaluation/validate_submission.py out/<team_name>.csv
```

The pipeline automatically downloads the three open Metrica training games into `devdata/` (if not already cached), trains the model with fixed random seeds (`seed=0`), decodes the challenge match in ~2 minutes on CPU, and outputs a byte-for-byte identical `out/<team_name>.csv` (`199,319` rows, `sha256: ee1320615ea0cab0`).