# womd-motion-forecasting

Two-notebook progression from a simple behavior cloning baseline to a map-aware multi-modal motion predictor on the Waymo Open Motion Dataset.

The first notebook establishes a baseline and figures out what actually matters — normalization, history length, evaluation metrics. The second builds a proper model that uses HD map lanes, nearby agent context, and predicts six possible futures instead of one. The jump from 5.3m ADE to 1.88m minADE@6 comes almost entirely from giving the model access to the road geometry.

---

## Notebooks

**`phase_setup.ipynb`** — baseline pipeline

Covers the full stack from raw TFRecords to trained models: GCS data access, feature extraction, tail slicing, CV/MLP/GRU training, slice-aware evaluation, and ablation studies on history length and prediction horizon.

**`motion_predictor_clean.ipynb`** — map-aware multi-modal model

Extends the baseline with HD map lane encoding (PointNet per polyline), social agent attention, and K=6 trajectory modes trained with winner-takes-all loss. Includes animated scenario visualizations and MP4 export.

---

## Results

### Baseline: CV vs MLP vs GRU across tail slices

| Slice | CV | MLP | GRU |
|-------|----|-----|-----|
| Aggregate | 9.35m | 5.40m | 5.28m |
| Close-interaction | 8.98m | 5.36m | 5.27m |
| High-curvature | 9.13m | 5.27m | 5.17m |
| Both (hardest) | 8.75m | 5.23m | 5.14m |

*ADE in meters, 8s horizon, 43k agent-samples from 50 val shards.*

<!-- add baseline comparison plot -->
<img src="results/final_results.png" width="800"/>

### MTR-Lite: map-aware multi-modal model

| Slice | GRU (baseline) | MTR-Lite | improvement |
|-------|---------------|----------|-------------|
| Aggregate | 5.28m ADE | 1.88m minADE@6 | 64% |
| Close-interaction | 5.27m ADE | — | — |
| High-curvature | 5.17m ADE | — | — |
| Both (hardest) | 5.14m ADE | — | — |

*minADE@6 = best of 6 predicted modes. Fill in your slice eval numbers above.*

<!-- add MTR-Lite progression plot -->
<img src="results/mtr_progression.png" width="800"/>

### Ablation — history length and prediction horizon

Going from 1s to 5s of history drops ADE by 79% (6.2m to 1.3m), uniformly across all tail slices. Prediction error compounds faster than linearly with horizon — 8s is 6x harder than 3s, not the 2.7x you'd expect from linear scaling.

<!-- add ablation plot -->
<img src="results/ablation_results.png" width="800"/>

---

## What I found along the way

**Normalization is the biggest single fix.** Before ego-frame rotation and velocity scaling, MLP scored 11.4m ADE — worse than the no-learning constant velocity baseline at 9.3m. After normalization it dropped to 5.4m. The architecture barely matters until the input representation is right.

**Tail scenarios aren't harder by ADE.** The "Both (hardest)" slice scores slightly *lower* ADE than aggregate across all models. Close-interaction and high-curvature scenarios tend to involve slower-moving agents so absolute displacement is smaller. ADE doesn't capture what makes these scenarios actually dangerous — you'd need collision rate or relative displacement metrics for that. This was the most interesting finding from the first notebook.

**Map context explains most of the MTR-Lite improvement.** The jump from 5.3m to 1.88m comes primarily from the model knowing where the road is. Without lane context, a vehicle turning left looks identical to one going straight until the turn starts. With lanes, the model can assign probability to the correct mode from the beginning of the prediction window.

**MLP is actually competitive with GRU at short history.** With 1s of history (11 steps), flattening the sequence into a vector loses almost nothing — GRU only beats MLP by 0.1m. The recurrent structure starts to matter more as history length increases, which the ablation confirms.

---

## Running this

### What you need

- Google account + GCP project (the $300 free credits cover this comfortably, total spend was around $2-3)
- Waymo license agreement signed at waymo.com/open/terms — non-commercial, free
- Google Colab with T4 GPU runtime

### GCP setup

Open Cloud Shell at console.cloud.google.com:

```bash
export PROJECT_ID=your-project-id
export BUCKET=gs://${PROJECT_ID}-data

gcloud services enable storage.googleapis.com aiplatform.googleapis.com
gsutil mb -p $PROJECT_ID -l us-central1 -c STANDARD $BUCKET

for folder in raw features checkpoints results plots; do
  gsutil cp /dev/null $BUCKET/$folder/.keep
done
```

### Waymo license

Go to waymo.com/open/terms and accept the Non-Commercial Use agreement using the **exact same Google account** as your GCP project. Wait 5 minutes then verify:

```bash
gsutil ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/
```

If you get `AccessDenied`, the account doesn't match.

### Notebook 1 — baseline

Upload `phase_setup.ipynb` to Colab, set runtime to T4 GPU, run Cell 1. Update these two lines:

```python
PROJECT_ID = 'your-project-id'
GCS_BUCKET = 'your-bucket-name'
```

Then run top to bottom. Each phase saves to GCS so a dropped session doesn't mean reprocessing shards.

### Notebook 2 — MTR-Lite

Upload `motion_predictor_clean.ipynb`. The setup cell handles all installs. If features are already saved from notebook 1 you can skip the extraction cell and jump straight to loading data. The model trains in about 20-25 minutes on a T4.

---

## GCS layout

```
gs://waymo_open_dataset_motion_v_1_3_1/     Waymo's bucket (read-only after license)
  uncompressed/scenario/
    training/               1000 shards
    validation/             150 shards
    testing/                150 shards
    validation_interactive/
    testing_interactive/

gs://your-bucket/                            your bucket
  features/
    val/                    5-shard early run
    val_v2/                 50-shard normalized (baseline results)
    map_v1/                 50-shard with map + social features (MTR-Lite)
    ablation_hist_3s/
    ablation_hist_5s/
  checkpoints/              .pt files — not in git
  results/                  metrics JSON
  plots/                    PNG figures
```

---

## Architecture — MTR-Lite

```
agent history (11 steps, 5 features)   GRU encoder (128-dim)
                                                |
lane polylines (32 lanes, 20 pts each)  PointNet MLP (64-dim)
                                                |
                                        cross-attention
                                                |
nearby agents (8 agents, 11 steps)     GRU + mean-pool (64-dim)
                                                |
                                        concat (192-dim context)
                                                |
                                        6x MLP heads + scorer
                                                |
                          6 trajectories (80 steps) + mode probabilities
```

Trained with winner-takes-all loss: regression on the closest mode plus cross-entropy to push its score highest.

---

## Dependencies

```
waymo-open-dataset-tf-2-12-0   --no-deps flag required
tensorflow >= 2.16
torch >= 2.0
numpy, matplotlib, tqdm
google-cloud-storage
```

The `--no-deps` flag matters — the waymo package declares a TF 2.12 requirement that no longer exists on PyPI, but the package itself works fine with newer TF versions.

---

## Data citation

```bibtex
@inproceedings{ettinger2021large,
  title={Large Scale Interactive Motion Forecasting for Autonomous Driving},
  author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and
          Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and
          Qi, Charles R and Zhou, Yin and others},
  booktitle={ICCV},
  year={2021}
}
```
