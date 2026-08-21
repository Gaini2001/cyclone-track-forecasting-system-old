"""
src

Cyclone track forecasting pipeline.

Packages are prefixed s1_ through s7_ so the directory listing reads in
execution order. The letter prefix is deliberate: a bare digit is not a valid
Python identifier, so `src.1_ingestion` cannot appear in an import statement
at all.

    s1_ingestion      download and schema validation
    s2_preprocessing  cleaning, filtering, segmentation
    s3_features       feature construction and targets
    s4_validation     temporal correctness and leakage checks
    s5_training       splitting, training, baselines, evaluation
    s6_inference      prediction and forecast tracks
    s7_analysis       exploratory and evaluation figures
    utils             config, logging, timing, metrics
"""
