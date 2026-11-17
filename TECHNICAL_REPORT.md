# Collective State-Space Possession Inference Under Severe Ball Occlusion

**More Than a Hack 2027 — Initial Challenge Technical Document**

**Team:** FalseNineAnalytics  
**Code:** https://github.com/Constifk/FalseNineAnalytics  
**Submission file:** `FalseNineAnalytics.csv` (199,319 rows), produced by `python -m possession.run --team FalseNineAnalytics --train metrica`

---

## 1. The Approach

### Problem Formulation
In the Initial Challenge (`FC Barcelona vs. Athletic Club`, 199,319 frames at 25 Hz), the central ranking metric is **`spellF1`** — temporal Intersection-over-Union (IoU $\in \{0.3, 0.4, 0.5, 0.6, 0.7\}$) over maximal contiguous runs of live possession (`H` or `A`), where dead-ball stretches (`D`) act as boundary separators between possessions.

Optimizing for `spellF1` differs fundamentally from optimizing per-frame accuracy (`macroF1`):
1. **Zero tolerance for label flicker:** A single spurious frame of `A` or `D` in the middle of an `H` possession splits one true possession into two false positives and destroys temporal IoU.
2. **Stoppage boundaries (`D`) are as critical as team switches:** Missing a stoppage merges two consecutive same-team possessions into one over-extended spell, costing both recall and precision.
3. **40.7% missing ball & anonymous tracklets:** The ball is absent on 40.7% of graded frames (disproportionately during duels, aerial balls, and stoppages), and players are represented by 46 anonymous, identity-swapping track slots (~20.5 active per frame).

### Three-Stage Architecture
We built a deterministic, zero-leakage pipeline (`possession/`) developed and cross-validated exclusively on **ten open matches from other competitions** (the three Metrica Sports sample games and the seven DFL games of the IDSSE dataset), degraded through a calibrated Automatic Tracking Data (ATD) noise simulator. The submitted model is trained on the three Metrica games (`--train metrica`, see Section 3 for why):

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
| **Tactical Smart Tagging (`BAR-ATH_tactical_smart_tagging.xml`)** | **Built, excluded from the default** | We built an unsupervised polarity calibrator (`possession/tagging.py`) that aligns phase codes with tracking-only possession estimates, and per-frame features for set-piece windows and phase team. They are excluded by default for reasons that do not depend on the provided game: (1) phase intervals are phases, not possessions — a phase can span one possession, stop short of one, or cover two (`DATA.md`); (2) the meaning of a phase's `Team` tag is not documented for every code, so its polarity has to be guessed; (3) the open training games have no Smart Tagging, so tag features can only be trained on our own simulation of tags (`tagging.simulate`), which encodes exactly those guesses. *Disclosure:* the question first came up when an early version of our run script also scored the public 5-minute window, with and without tags. That window is not a tuning set, and 17 possessions are far too few for a difference to be evidence, so the decision rests on (1)–(3); the run script no longer scores the window at all. Tags remain available via `--use-tags`. |
| **Broadcast ATD & Video** | **Not used** | The tactical ATD covers 100% of frames and is the reference timeline, whereas the broadcast feed is not synchronised and has usable homography on 65.8% of its frames (`DATA.md`). We did not test it. |
| **Tactical Homography** | **Not used** | The tactical ATD is already in pitch coordinates. The homography would only be needed for a video-based ball detector (Section 5). |

---

## 3. Training Data & Label-Free ATD Noise Calibration

### Compliance summary (`RULES.md` and the organisers' clarification email of 3 November 2026)

| Rule | How we comply |
|---|---|
| No hand labels, human-produced data or manually created data on the provided game | None used. Every label in the submission comes out of `possession.run`. Nobody labelled or corrected any frame of the provided game. |
| No training, fitting or calibrating on the ground truth or on the provided video | The classifier weights are fit **only on external games** (the three Metrica games for the submission) and the Viterbi costs were selected on the ten external games. The public 5-minute ground truth is never read by `possession.run`; the video is not an input. |
| Tuning only on external data | The only free parameters (Viterbi costs, regularisation) are grid-searched on the external games (leave-one-game-out, Section 3). |
| No checking of the output against the video | Not done. |
| No upload of the provided data to AI services | The provided data never left the team's own machines. AI coding assistants were used on the code only; the repository ships exclusion files (`.claude/settings.json`, `.cursorignore`, `.geminiignore`, `.aiexclude`) for `data/` and `out/`. |
| Licensed external data | Metrica sample data (repository licence) and IDSSE/DFL (CC BY 4.0), each pinned by commit and sha256. |

*Disclosure on the noise simulator.* The synthetic degradation applied to the external games was matched to **label-free aggregate statistics of the provided tracking data** (ball-gap frequency and duration quantiles, player-speed quantiles; `possession/diagnose.py`). No ground truth, no video and no frame-level information enters this step, and the model itself is still fit only on external games. We list it explicitly because it is the one place where the provided game informs a setting, so the organisers can judge it.

### Details

In strict compliance with `RULES.md`:
- **No labels from the provided game** were used for training, hyperparameter selection, or threshold tuning. The public 5-minute window was used only to check that frames are aligned and the file is well-formed (`evaluation/validate_submission.py`, `evaluation/run_evaluation.py` on the official baselines); `possession.run` does not score it.
- **The only information taken from the provided game is label-free:** aggregate kinematic distributions of its tracking data, used to calibrate the noise simulator (below). `RULES.md` permits learning without human labels.
- **Development corpus (10 open games from other competitions; the submission model is trained on the first three):**
  - The three Metrica Sports sample games (`github.com/metrica-sports/sample-data`, pinned to commit `e706dd5`). Ground-truth `H`/`A`/`D` labels are derived mechanically from the open event streams following the rules in `CHALLENGE.md` (`possession/devgames.py`).
  - The seven DFL games (two Bundesliga, five 2. Bundesliga, 2022/23) of the IDSSE dataset: Bassek, Rein, Weber & Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195, doi:10.1038/s41597-025-04505-y, licensed **CC BY 4.0**. Downloaded from the PySport mirror `huggingface.co/datasets/pysport/idsse-data` at a pinned revision (`possession/idsse.py`). Their position feed carries the tracking provider's `BallStatus` and `BallPossession` on every frame, so the labels are derived the way the challenge ground truth is: mechanically from ball status and possession flags, without smoothing (`D` where the ball is dead, otherwise the team in possession).
  - Every downloaded file is checked against a pinned sha256.
- **Realistic ATD Noise Simulation (`possession/noise.py`):** Clean open tracking data differs sharply from raw computer-vision ATD. Using a label-free diagnostic (`possession/diagnose.py`) that compares only aggregate kinematic distributions (gaps per minute, gap duration quantiles, and player speed quantiles), we discovered two critical physical properties of real ATD:
  1. **Ball dropouts occur in long contiguous bursts** (1.4 gaps/min, median gap 10.7 s, 90th percentile 33.8 s) rather than sub-second flickers.
  2. **Player trajectories are temporally smooth** (median player speed $1.54\text{ m/s}$, 10th percentile $0.71\text{ m/s}$ during stoppages). Over-jittering synthetic player positions inflates resting player speeds above $1.6\text{ m/s}$ and blinds the stoppage (`D`) detector.
  Calibrating `NoiseConfig` (`gap_median_frames=250`, `player_jitter_m=0.04`, across three presets `mid`, `alive_hard`, `dead_dark`) matched these label-free distributions (`python -m possession.diagnose` prints both side by side).

### Leave-One-Game-Out Benchmark (10 open games)

`python -m possession.bench --quick`: each of the ten open games is held out once; the model is trained on the other nine, all degraded with the calibrated `mid` noise preset, and the Viterbi costs are grid-searched on the training games only (grid `{8, 32}` for each cost). All ten folds select the same costs (`switch=8`, `stop=32`, `start=32`), which are the ones used for the submission.

| Method | Metrica games (3) `spellF1` | DFL games (7) `spellF1` (min) | DFL `macroF1` | DFL predicted / true possessions | Description |
|---|---|---|---|---|---|
| `v0` (Baseline reproduction) | `0.293` | `0.246` (`0.225`) | `0.651` | 3,161 / 3,061 | Nearest player + $\pm 2\text{ s}$ majority vote (`baseline_smooth50`) |
| `v1` (Rule-based memory) | `0.378` | `0.320` (`0.249`) | `0.729` | 1,914 / 3,061 | Ball-control radius ($1.5\text{ m}$, $\ge 5$ frames) + collective slowdown stoppage rule |
| **`v2` (Collective LogReg + Viterbi)** | **`0.508`** | **`0.398`** (`0.351`) | **`0.832`** | 1,811 / 3,061 | **29 collective features + prior-corrected multinomial emissions + Viterbi decoding** |

Two observations. First, the baseline reproduction scores `0.246` on the degraded DFL games, close to the official baseline's `0.2433` on the provided game, so the DFL games under our noise model are a fair stand-in; the Metrica games are easier. Second, adding the DFL games to training did not move the mean over all ten held-out games (`0.431` trained on the Metrica games only, `0.431` trained on all; `python -m possession.bench --quick --train metrica`): the model is limited by its features, not by the amount of data. Because the two training pools are equivalent on all ten held-out games, the submitted file is produced with the smaller pool (`--train metrica`, ~200 MB of downloads instead of ~2.8 GB), which makes reproduction much faster; `--train all` remains available. This choice was made on the open-game benchmark only, never on the provided game. `v2` predicts only about 60% of the true number of possessions on the DFL games: it merges possessions across missed turnovers and short stoppages (Section 5).

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

- **Too few possessions:** On the held-out games `v2` predicts 75% of the true number of possessions on the Metrica games (799 of 1,064) and 59% on the DFL games (1,811 of 3,061), so turnovers and short stoppages are missed and two possessions merge into one.
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
.venv\Scripts\python -m possession.run --team FalseNineAnalytics --train metrica
.venv\Scripts\python evaluation/validate_submission.py out/FalseNineAnalytics.csv
```

On Windows, `ABGABE-ERSTELLEN.bat` runs the same steps end to end (Python 3.12 setup, data placement via `possession.unpack`, run, official validation).

With `--train metrica` the pipeline downloads the three Metrica training games into `devdata/` on first use (about 200 MB, each file checked against a pinned sha256; `--train all` adds the seven DFL games, about 2.8 GB), trains the model with fixed seeds, decodes the challenge match in a few minutes on CPU, and writes `out/FalseNineAnalytics.csv` (`199,319` rows) together with its sha256.

**Byte-identical across machines.** BLAS and numpy's SIMD kernels differ between CPUs in the last bits; we measured differences of `1e-15` in the learnt weights between OpenBLAS kernels. The learnt parameters are therefore rounded to `1e-9` and the emission scores to `1e-6` before decoding, and the decoder is a scalar Python loop. On a stand-in bundle built from an open game (`python -m possession.fake_bundle`) the output is identical with the default kernels, with a forced old kernel (`OPENBLAS_CORETYPE=Prescott`, AVX2/AVX-512 disabled in numpy) and with different thread counts. Python 3.12 and the pinned `requirements-team.txt` are assumed.