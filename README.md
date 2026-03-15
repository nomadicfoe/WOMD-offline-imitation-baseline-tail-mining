# womd-motion-baseline

Behavior cloning baseline for motion forecasting on the Waymo Open Motion Dataset, with tail-scenario mining to find where models actually fail.

I built this to understand what a "real" motion forecasting pipeline looks like from scratch — starting with raw WOMD TFRecords on GCS, building up feature extraction, training CV/MLP/GRU baselines, and then running slice-based evaluation to see whether models fail more on close-interaction and high-curvature scenarios than on average. Spoiler: ADE doesn't actually capture tail difficulty well, which turned out to be the most interesting finding.

---

## What this covers

- Streaming WOMD v1.3.1 TFRecords directly from GCS in Colab
- Tail slicer — two criteria: close-interaction (agents within 5m) and high-curvature (turning radius < 20m)
- Three baselines: constant velocity (no learning), MLP (flatten history), GRU (sequential)
- Ego-frame normalization — turned out to be the single most important preprocessing step
- Slice-aware eval — ADE/FDE broken out per tail slice, not just aggregate
- Ablations on history length (1s / 3s / 5s) and prediction horizon (3s / 5s / 8s)

The notebook is one file, runs end-to-end in Colab, and saves features + checkpoints to GCS so you don't have to re-process shards every session.

---

## Results

### Model comparison

| Slice | CV | MLP | GRU |
|-------|----|-----|-----|
| Aggregate | 9.35m | 5.40m | 5.28m |
| Close-interaction | 8.98m | 5.36m | 5.27m |
| High-curvature | 9.13m | 5.27m | 5.17m |
| Both (hardest) | 8.75m | 5.23m | 5.14m |

*ADE in meters, 8s horizon, 43k agent-samples from 50 val shards.*

<!-- drop your plot here -->
<img src="results/final_results.png" width="800"/>

### Ablation — history length vs prediction horizon

The history length result surprised me. Going from 1s → 5s of history drops ADE by 79% (6.2m → 1.3m), and the improvement is almost identical across tail and aggregate slices. Prediction horizon error compounds much faster than linearly — 8s is 6× harder than 3s, not 2.7×.

<!-- drop your ablation plot here -->
<img src="results/ablation_results.png" width="800"/>

---

## Things I found interesting

**Normalization matters more than architecture.** Before adding ego-frame rotation + velocity scaling, MLP was actually worse than the no-learning constant velocity baseline (11.4m vs 9.3m ADE). After normalization it dropped to 5.4m. The GRU only beats MLP by ~0.1m — with 1s history, flattening the sequence barely loses anything.

**Tail scenarios aren't harder by ADE.** The "Both (hardest)" slice consistently scores slightly lower ADE than aggregate. This is because close-interaction and high-curvature scenarios tend to involve slower agents — absolute displacement is smaller. If you want to actually stress-test tail performance, you'd need collision rate or relative displacement metrics, not ADE.

**History length is the dominant knob.** More history helps uniformly across all slices. Longer horizons just make everything harder at the same relative rate across slices.

---

## Running this yourself

### What you need

- A Google account + GCP project (the $300 free credits are more than enough — this whole project cost ~$2)
- Waymo license agreement signed (free, takes 5 minutes)
- Google Colab (free T4 GPU tier is fine)

### Step 1 — GCP project setup

If you're starting from scratch, open Cloud Shell at [console.cloud.google.com](https://console.cloud.google.com) and run:

```bash
export PROJECT_ID=your-project-id
export BUCKET=gs://${PROJECT_ID}-data

gcloud services enable storage.googleapis.com aiplatform.googleapis.com
gsutil mb -p $PROJECT_ID -l us-central1 -c STANDARD $BUCKET

# create folder structure
for folder in raw features checkpoints results plots; do
  gsutil cp /dev/null $BUCKET/$folder/.keep
done
```

### Step 2 — Sign the Waymo license

Go to [waymo.com/open/terms](https://waymo.com/open/terms) and accept the Non-Commercial Use agreement. Use the **exact same Google account** as your GCP project — this is the only thing that can go wrong here and it's easy to miss.

After signing, wait about 5 minutes then verify:

```bash
gsutil ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/
```

You should see the training/validation/testing splits listed. If you get `AccessDenied`, double-check the account matches.

### Step 3 — Open the notebook

Upload `phase_setup.ipynb` to Colab, switch runtime to **T4 GPU**, and run Cell 1. It installs deps and authenticates via OAuth. Update these two lines in Cell 3 to match your setup:

```python
PROJECT_ID = 'your-project-id'
GCS_BUCKET = 'your-bucket-name'
```

Then run top to bottom. Each phase saves outputs to GCS so if the session drops you can pick up without reprocessing.

---

## GCS layout

```
gs://waymo_open_dataset_motion_v_1_3_1/   ← Waymo's bucket (read-only after license)
  uncompressed/scenario/
    training/        1000 shards
    validation/      150 shards
    testing/         150 shards
    validation_interactive/
    testing_interactive/

gs://your-bucket/                          ← your bucket (read/write)
  features/
    val/             early 5-shard run
    val_v2/          50-shard normalized features (used for final results)
    ablation_hist_3s/
    ablation_hist_5s/
  checkpoints/       model .pt files (not in git — too large)
  results/           metrics JSON
  plots/             PNG figures
```

---

## Dependencies

```
waymo-open-dataset-tf-2-12-0   install with --no-deps
tensorflow >= 2.16
torch >= 2.0
numpy, pandas, matplotlib, seaborn, tqdm
google-cloud-storage, protobuf
```

```python
!pip install -q waymo-open-dataset-tf-2-12-0 --no-deps
!pip install -q tensorflow numpy pandas matplotlib seaborn tqdm protobuf
```

The `--no-deps` flag on the waymo package is important — it ships with a TF 2.12 requirement that no longer exists on PyPI, but the package itself works fine with TF 2.16+.

---

## What's next

This is the v1 baseline — agents predicted in a vacuum with no map awareness. The follow-up notebook (`02_map_aware_motion_transformer`) adds HD map lane encoding, multi-agent social attention, and K=6 multi-modal outputs, targeting ~1.5m minADE@6 vs the 5.3m here.

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
