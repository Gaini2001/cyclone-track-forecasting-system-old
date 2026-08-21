# Cyclone Track Forecasting

Forecasting tropical cyclone positions from NOAA IBTrACS best-track data, at
lead times from 6 to 72 hours.

The interesting part of this project is not the model. It is that the first
version of it lost to a three-line physical baseline, and finding out why
required rebuilding the evaluation before rebuilding the model.

---

## Headline result

At a 24-hour lead time, on held-out seasons:

| Forecast             | Mean track error (km) | Skill vs persistence | Skill vs linear extrapolation |
| :------------------- | --------------------: | -------------------: | ----------------------------: |
| XGBoost              |                  116.4 |                +71.2% |                         +27.3% |
| Random Forest        |                  120.0 |                +70.3% |                         +25.0% |
| Linear extrapolation |                  160.1 |                +60.3% |                              — |
| Persistence          |                  403.4 |                     — |                              — |

Produced by `python -m src.s5_training.evaluate --horizon all`, which writes the
full table to `reports/results.md` with bootstrap confidence intervals.

![Track error by lead time](reports/figures/error_vs_lead_time.png)

**Report skill against linear extrapolation, not persistence.** Persistence
assumes the storm stops moving, which is easy to beat and gets easier as the
lead time grows — quoting it flatters a model without saying anything.
Constant-velocity extrapolation is the reference that tests whether a model has
learned steering rather than inertia.

---

## What went wrong the first time

The original pipeline reported a 6-hour mean track error of 30.16 km and an
R² of 0.9998, and lost to linear extrapolation (13.76 km) by a factor of two.
Both facts had the same cause.

**The target was the absolute future coordinate**, while the current
coordinate was an input feature. The model's job reduced to copying its input
forward, which produces an R² near 1.0 and tells you nothing. Impurity
importance on the original Random Forest:

| Feature group                                |       Importance |
| :------------------------------------------- | ---------------: |
| Position (LAT/LON and lags)                  | **95.70%** |
| Intensity (wind, pressure, lags)             |            1.91% |
| Temporal and cyclic                          |            1.44% |
| Context (speed, direction, distance to land) |            0.71% |
| Motion (deltas, distance, bearing)           |            0.24% |
| Acceleration                                 |            0.00% |

Longitude and its three lags alone accounted for 89.8%. Every physically
motivated feature was contributing nothing, because with an absolute target
there was nothing left for them to explain. Copying position is worse than
copying velocity, which is why a straight-line baseline won.

Four further defects were found while investigating:

| Defect                                     | Effect                                                                                                   |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| `bfill` imputation within storms         | Copied later observations backwards into rows used as model inputs — future information in the features |
| Horizons defined as row shifts             | IBTrACS interpolates to 3-hourly rows, so a "6-hour target" was sometimes 3 hours ahead                  |
| `BASIN == "NA"` parsed as null           | pandas treats`"NA"` as missing; every North Atlantic storm silently lost its basin label               |
| Storms crossing the antimeridian discarded | Removed most of the West Pacific, the busiest basin, to avoid a longitude wrap                           |

---

## What changed

**Displacement targets.** The model predicts (Δlat, Δlon) from the issue-time
position rather than the position itself, so R² measures forecast skill instead
of self-similarity.

**Verified time alignment.** Targets are checked against `ISO_TIME` by a
self-join rather than by re-applying the shift that produced them, so a row
offset that does not correspond to the stated number of hours fails the build.
Data is filtered to synoptic hours (00/06/12/18) and to `TRACK_TYPE == "main"`.

**Contiguous track segments.** Cleaning removes rows, which leaves storms with
holes. A `SEGMENT_ID` marks runs of strictly 6-hourly observations; lags and
targets group by segment, while train/test splits group by storm.

**Season-based holdout.** Training on earlier seasons and testing on later ones,
which is how a forecast system is actually used. The previous random partition
of storm IDs trained on 2021 and tested on 1987.

**Causal imputation only.** Forward-fill, capped at two steps. Position is never
imputed — a filled position fabricates a stationary storm and corrupts both the
motion features and the persistence baseline.

**Wrapped longitude arithmetic.** Differences are wrapped to (−180, 180], so
storms crossing the antimeridian are kept rather than discarded.

**New features.** Turning rate (the recurvature signal), velocity components in
km/h, rolling 24-hour motion, basin encoding, and circular sin/cos encodings for
bearing and storm direction. 37 engineered features plus identifiers and lags
became 56 columns in the final feature matrix.

---

## Evaluation

**Along-track and cross-track decomposition.** Aggregate error says how wrong a
forecast was; the decomposition says how. Cross-track error is perpendicular to
the storm's motion — a direction error, typically a missed recurvature.
Along-track error is parallel — a speed error, and a non-zero mean is a
systematic bias.

![Error decomposition](reports/figures/along_cross_track.png)

At 24 hours the model's cross-track MAE is 70.1 km against an along-track
bias of -36.7 km. The two failure modes are of comparable size, and the
negative along-track bias — under half of the 77.6 km along-track MAE — shows
the model systematically under-runs storm translation speed rather than being
dominated by one error type.

**Bootstrap confidence intervals** accompany every skill claim, via a paired
test on the same test rows. With 216 test storms, a difference smaller than
its interval is not a result.

**Feature ablation.** Each feature group is removed and the model retrained,
which is the only way to answer "would I lose anything by deleting these?"
Permutation importance cannot: shuffle `DELTA_LON` and the model leans on
`VELOCITY_U` instead.

| Group removed  | Change in track error (km) |
| :------------- | --------------------------: |
| velocity       |                       +14.54 |
| intensity      |                       +12.53 |
| rolling        |                        +0.86 |
| motion         |                        +0.47 |
| acceleration   |                        +0.36 |
| turning        |                        +0.25 |
| position       |                        +0.16 |
| context        |                        +0.09 |
| lag_position   |                        +0.05 |
| temporal       |                        +0.04 |
| lag_intensity  |                        -0.04 |
| cyclic         |                        -0.13 |

Velocity and intensity are the only groups that matter at 24h: losing either
costs 12-15 km. Everything else is inside noise, including two groups
(`lag_intensity`, `cyclic`) whose removal nominally *helped* — a difference
this small is the ablation study's way of saying "delete these and check
nothing else breaks," not "these hurt the model."

**Forecast tracks.** One issue time projected to every lead time, verified
against the observed track.

![Forecast track](reports/figures/forecast_track_example.png)

---

## Dataset

NOAA IBTrACS v04r00, all basins, 1980 onwards. Place the CSV at
`data/raw/ibtracs.csv` before running the pipeline.

|                                |             |
| :----------------------------- | ----------: |
| Observations after cleaning    |     62,206 |
| Storms                         |      1,933 |
| Contiguous segments            |      1,951 |
| Seasons                        | 1980-2025 |
| Retention through the pipeline |    20.59% |

From `python -m src.s7_analysis.eda`. Retention is well under 100% by design:
off-synoptic rows, spur tracks, non-tropical stages, and the head and tail of
every segment are all removed deliberately.

![Dataset composition](reports/figures/dataset_composition.png)

---

## Pipeline

```
IBTrACS CSV
    │  src/s1_ingestion/dataset_loader.py     download, schema check, Parquet cache
    ▼
Raw observations
    │  src/s2_preprocessing/clean_data.py     main tracks, synoptic hours, causal
    │                                        imputation, segment assignment
    ▼
Clean observations
    │  src/s3_features/feature_engineering.py lags, motion, velocity, turning,
    │                                        rolling, context, targets
    ▼
Feature matrix
    │  src/s4_validation/validate_features.py time alignment, lag correctness,
    │                                        NaN placement, leakage sentinel
    ▼
Validated features
    │  src/s5_training/train.py               season split, fingerprinted cache
    │  src/s5_training/train_model.py         RF + XGBoost, per horizon
    ▼
Models  ──────────────────────────────────┐
    │  src/s5_training/evaluate.py         │  app/main.py     FastAPI
    ▼                                     └─ app/app.py      Streamlit
reports/results.md, reports/figures/
```

---

## Running it

```bash
pip install -r requirements.txt

python main.py                    # full pipeline
```

Or stage by stage:

```bash
python -m src.s2_preprocessing.clean_data
python -m src.s3_features.feature_engineering
python -m src.s4_validation.validate_features
python -m src.s5_training.tune_hyperparameters --model all --horizon all --trials 50
python -m src.s5_training.train_model --model all --horizon all
python -m src.s5_training.evaluate --horizon all
python -m src.s7_analysis.eda
python -m src.s7_analysis.feature_analysis --horizon 24 --ablation
```

Inference:

```bash
python -m src.s6_inference.predict --list-storms
python -m src.s6_inference.predict --sid <SID>          # forecast track
python -m src.s6_inference.predict --export             # per-row predictions
```

Serving:

```bash
uvicorn app.main:app --reload --port 8000     # http://localhost:8000/docs
streamlit run app/app.py --server.port 8501
```

Tests:

```bash
pytest tests/ -v                  # 112 tests
REQUIRE_API=1 pytest tests/ -v    # fail rather than skip if the API is down
```

Docker:

```bash
docker compose up --build
```

---

## Notes on reproducibility

- Every random seed derives from `RANDOM_STATE` in `src/utils/config.py`.
- Split membership is persisted under a filename fingerprinted by the split
  configuration, so a run with different ratios cannot silently reuse another
  run's split.
- Each model artifact carries a JSON sidecar recording its feature names and
  order, hyperparameters, split fingerprint, and the library versions that
  produced it. Loading a model whose feature set no longer matches raises
  rather than predicting.
- Tuned hyperparameters are written to `reports/tuned_params.json` by the
  Optuna study and read back automatically — nothing is copied by hand.
- Model artifacts are not committed. Train them, or the API reports `degraded`.

---

## Limitations

- **Best-track data is a reanalysis, not observations.** IBTrACS positions are
  post-season best estimates informed by the storm's full history. A model
  trained on them will do better than one trained on what a forecaster had at
  the time.
- **No atmospheric fields.** Operational track forecasting is driven by steering
  flow from numerical weather prediction. This model sees only the storm's own
  history, which bounds how well it can anticipate recurvature.
- **Intensity is not forecast**, so the recursive rollout in
  `src/s6_inference/predict.py` holds wind and pressure constant. The direct
  strategy — one model per lead time — is the one to rely on.
- **Not comparable to official forecasts.** NHC and JTWC verification uses
  different samples, different basins, and homogeneous comparison sets. The
  numbers here are internally consistent and not a like-for-like benchmark.

---

## Author

**Om Prakash Gaini**
M.Tech, AI for Sustainability — Indian Institute of Technology Kanpur

## License

This project is licensed under the MIT License.
