# Increasing-Extreme-Negative-Vegetation-Responses-to-Drought-Related-Compound-Events
This repository contains the tools and algorithms for identifying, attributing, and analyzing extreme vegetation anomaly events.

## `tools/`

Contains general-purpose utility functions used throughout the analysis, including:

- Binary encoding and decoding functions for converting between binary lists and integers.
- Functions for calculating the actual area of regular latitude–longitude grid cells.

## `Extreme_Event_Detection_Algorithm/`

Contains the algorithms for identifying extreme events.

The `example/` folder provides examples of how to use the event detection functions to:

- Identify spatially and temporally connected vegetation anomaly events.
- Convert climate anomalies into binary (0/1) event matrices for subsequent attribution analysis.

## `attribution_analysis/`

Contains the code for attributing vegetation anomaly events to different climate anomalies.

### `shuffle/`

Performs 1,000 shuffle iterations to generate the null distribution of the overlap between climate anomalies and vegetation anomaly events.

### `coincidence_analysis/`

Calculates the observed overlap rate and the overlap rates obtained from the 1,000 shuffled climate anomalies. These results are used to determine whether a climate anomaly can be identified as a significant driver of a vegetation anomaly event under a specified *p*-value threshold.

### `pickle_class/`

Packages and organizes the attribution results.

For each vegetation anomaly event, a unique key is assigned and linked to an *N*-bit binary list. Each bit represents one type of climate anomaly:

- `1` indicates that the corresponding climate anomaly is identified as a significant driver of the vegetation anomaly event.
- `0` indicates that the corresponding climate anomaly is not identified as a significant driver.

## Overall Workflow

The overall workflow can be summarized as:

```text
Vegetation Data
      │
      ▼
Extreme Event Detection
      │
      ▼
Connected Vegetation Anomaly Events
      │
      ├───────────────┐
      ▼               ▼
Climate Anomalies   Shuffle Analysis
      │               │
      └───────┬───────┘
              ▼
     Coincidence Analysis
              │
              ▼
    Event Attribution
              │
              ▼
    Binary Attribution Results
