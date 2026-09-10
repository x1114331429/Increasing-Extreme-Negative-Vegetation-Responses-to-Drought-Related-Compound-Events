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

## Data Preparation

Before running the analysis, the required input datasets should be prepared in advance. All gridded datasets should use the same spatial resolution and spatial extent.

### 1. Vegetation, Climate, and Burned Area Data

Vegetation data, climate data, and burned area data should be provided as monthly time-series arrays with the following shape:

```text
(360, 720, n × 12)
```

where:

- `360 × 720` represents the global 0.5° × 0.5° spatial grid.
- `n` represents the number of years.
- `n × 12` represents the monthly time dimension.

For example, a 24-year monthly dataset should have a shape of:

```text
(360, 720, 288)
```

The same data structure should be used for vegetation variables, climate variables, and burned area data.

Examples of vegetation variables include:

- LAI
- NDVI

Examples of climate variables include:

- Temperature
- Precipitation
- Soil moisture
- VPD

Burned area data should also be provided at monthly resolution using the same spatial and temporal dimensions.

### 2. Land-use Data

Land-use data should be provided at annual resolution with the following shape:

```text
(360, 720, n)
```

where:

- `360 × 720` represents the global 0.5° × 0.5° spatial grid.
- `n` represents the number of years.

Unlike vegetation, climate, and burned area data, the temporal dimension of land-use data does not need to be multiplied by 12 because land-use data are provided annually.

### 3. Growing Season Data

A growing-season matrix should be prepared to identify whether each grid cell is within the growing season for each month.

The growing-season matrix should have the following shape:

```text
(360, 720, n × 12)
```

The values should be binary:

- `1`: the grid cell is within the growing season for that month.
- `0`: the grid cell is outside the growing season for that month.

The growing-season matrix should have the same spatial and temporal dimensions as the corresponding monthly vegetation and climate datasets.

### 4. Spatial Mask

A spatial mask should be provided to define the region of interest.

The mask should have the following shape:

```text
(360, 720)
```

The mask should contain binary values:

- `1`: the grid cell is included in the analysis.
- `0`: the grid cell is excluded from the analysis.

For example, in this study, the mask represents the land vegetation area.

The spatial mask should use the same spatial grid as all other input datasets.
