# Time Series Classification - Feature Creation and Extraction
Vu Nguyen - USC ID: 2120314402

## Introduction

This project involves classifying human activities based on time series data obtained from a Wireless Sensor Network. The dataset used for this project is the AReM dataset, which contains multiple instances of human activities, each recorded as a time series of sensor data.

## Dataset Description

### Source
The dataset can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29).

### Structure
The dataset consists of 7 folders, each representing a different type of activity:

1. **bending1**
2. **bending2**
3. **cycling**
4. **lying**
5. **sitting**
6. **standing**
7. **walking**

Each folder contains multiple CSV files, where each file corresponds to an instance of a person performing the specified activity. Each CSV file contains six time series recorded from the same person:

- `avg rss12`
- `var rss12`
- `avg rss13`
- `var rss13`
- `avg rss23`
- `ar rss23`

In total, there are 88 instances across all activities, with each time series containing 480 consecutive values.

## Data Splitting

### Train/Test Split

- For the activities `bending1` and `bending2`, the first two instances are used as test data.
- For all other activities, the first three instances are used as test data.
- The remaining instances in each folder are used as training data.

## Feature Extraction

### Time-Domain Features

In time series classification, it is common to extract various statistical features that represent the characteristics of the data. The following time-domain features were extracted for each of the six time series in every instance:

1. **Minimum (`min`)**: The smallest value in the time series.
2. **Maximum (`max`)**: The largest value in the time series.
3. **Mean (`mean`)**: The average value of the time series.
4. **Median (`median`)**: The middle value of the time series when sorted.
5. **Standard Deviation (`std`)**: A measure of the dispersion or spread of the time series values.
6. **First Quartile (`q1`)**: The 25th percentile value of the time series.
7. **Third Quartile (`q3`)**: The 75th percentile value of the time series.

These features provide a summary of the central tendency, variability, and distribution of the time series data.

### Extracted Features Format

After feature extraction, the dataset is transformed into a new format with the following structure:

| Instance | min1 | max1 | mean1 | median1 | std1 | q11 | q31 | ... | min6 | max6 | mean6 | median6 | std6 | q16 | q36 |
|----------|------|------|-------|---------|------|-----|-----|-----|------|------|-------|---------|------|-----|-----|
| 1        | ...  | ...  | ...   | ...     | ...  | ... | ... | ... | ...  | ...  | ...   | ...     | ...  | ... | ... |
| 2        | ...  | ...  | ...   | ...     | ...  | ... | ... | ... | ...  | ...  | ...   | ...     | ...  | ... | ... |
| ...      | ...  | ...  | ...   | ...     | ...  | ... | ... | ... | ...  | ...  | ...   | ...     | ...  | ... | ... |
| 88       | ...  | ...  | ...   | ...     | ...  | ... | ... | ... | ...  | ...  | ...   | ...     | ...  | ... | ... |

Each row in the table represents an instance, and the columns represent the extracted features for each of the six time series.

## Statistical Analysis

### Standard Deviation and Confidence Intervals

For each extracted feature, the standard deviation was calculated to assess the variability across instances. To provide an estimate of the precision of these variability measures, a 90% bootstrap confidence interval was constructed for each feature's standard deviation. This involves repeatedly resampling the data with replacement and calculating the standard deviation for each sample.

### Bootstrap Confidence Interval
The 90% confidence interval provides a range within which the true standard deviation of the feature is likely to fall, with 90% confidence.

## Feature Selection

After evaluating the statistical properties of each feature, three key time-domain features were selected based on their relevance to capturing the characteristics of the time series data:

1. **Mean (`mean`)**: Represents the central tendency of the time series data, providing a general sense of the activity's intensity or level.
2. **Standard Deviation (`std`)**: Measures the variability of the time series data, indicating how much the signal fluctuates around the mean.
3. **Maximum (`max`)**: Captures the peak values in the time series, highlighting extreme behaviors or outliers during the activity.

These features were chosen as they collectively provide a balanced view of the central tendency, variability, and extremes of the time series data, which are critical for distinguishing between different human activities.

## Project Structure

. ├── data
│   ├── bending1
│   │   ├── train
│   │   │   ├── dataset1.csv
│   │   │   └── ...
│   │   ├── test
│   │   │   ├── dataset2.csv
│   │   │   └── ...
│   ├── bending2
│   │   ├── train
│   │   ├── test
│   └── ...
├── split_dataset
│   ├── bending1
│   │   ├── train
│   │   ├── test
│   ├── bending2
│   │   ├── train
│   │   ├── test
│   └── ...
├── notebook
│   ├── Nguyen_Vu_HW3
└── README.md


- **data/**: Contains the original dataset structured in folders representing different activities.
- **split_dataset/**: Contains copies of the original dataset split into `train` and `test` subfolders for each activity.
- **scripts/**: Python scripts for feature extraction and dataset organization.
- **README.md**: Documentation for the project.

## Conclusion

This project extracted meaningful time-domain features from the AReM dataset and conducted a statistical analysis to understand the variability and reliability of these features. The selected features can be used for building classification models to accurately predict human activities based on time series data from a wireless sensor network.


