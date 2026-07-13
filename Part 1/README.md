Note : For few logics  referred offical documenation for syntax:
https://sqlite.org/lang.html
https://pandas.pydata.org/docs/
Project topic:

Predicting environmental air quality (AQI) and classifying air quality risk using meteorological conditions and large-scale climate variability (ENSO).

Problem Statement:
Air quality is influenced by multiple environmental and climatic factors rather than a single variable. This project aims to predict the Air Quality Index (AQI) using meteorological conditions and large-scale climate variability. In addition to predicting the AQI (regression), the project classifies air quality into Low Risk and High Risk categories (binary classification), enabling a simple environmental risk assessment.

Phase 0 – Solution Design

Before implementation, the project requirements, available datasets, and machine learning objectives were carefully evaluated. An initial idea of incorporating disease surveillance data was explored but later excluded because an official, publicly accessible dataset suitable for the project was not available. The project was therefore redesigned to focus on environmental risk prediction using reliable public datasets.

The final solution integrates three authoritative datasets:

**Air Quality Data**: Air Quality data obtained from Kaggle, released under CC0 1.0 Universal Public Domain Dedication."
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

**Weather Data**:This project utilizes Indian Rainfall and Weather Prediction Dataset used under CC BY-NC 4.0. Data was modified by merging it with external datasets to form the project architecture(https://www.kaggle.com/datasets/ameydilipmorye/indian-rainfall-and-weather-data)

**ENSO (Oceanic Niño Index)**: National Oceanic and Atmospheric Administration (NOAA)(https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)
These datasets were selected because they are publicly accessible, authoritative, and directly support the project objective.

Data Collection and Integration:
In Data folder source file,reduced csv and cleaned csv is present below are the data sources
The project combines air quality observations, meteorological measurements, and ENSO climate data into a single analytical dataset.

AQI data provides the prediction target and pollutant measurements.
Weather data provides environmental variables such as temperature, wind speed, rainfall, and atmospheric pressure.
ENSO data provides a large-scale climate indicator representing El Nino, La Nina, and Neutral conditions.

Since these datasets originated from different sources, they were integrated using SQL joins based on common attributes such as date and location before importing the consolidated dataset into Python. SQL was used because it is well suited for relational data integration,

 merged dataset contains approximately 27,000 records covered multiple Indian cities between 2015 and 2020., satisfying the assignment requirement of using a publicly available structured dataset with more than 500 observations.
Dataset Description. An initial data-quality assessment indicated that earlier periods contained substantially higher levels of missing values across several important variables. Instead of removing cities, which would reduce the geographical diversity and variability of the dataset, a quality-based date range with better data completeness was selected(2018 to 2020). This approach preserved observations from all cities while improving the overall quality of the dataset used for preprocessing and model development. All subsequent analyses in Part 1 were performed on this refined dataset.


Features Used:

City
Date
Season
AQI
AQI Bucket
Average Temperature
Minimum Temperature
Maximum Temperature
Wind Speed
Air Pressure
Oceanic Niño Index (ONI)


Task 1 — Dataset Loading:
Loaded the dataset using pandas.read_csv()
Verified dataset dimensions
Checked data types
Displayed the first five records

Task 2 — Missing Value Analysis

Missing value analysis was performed on the dataset to assess data quality before preprocessing. The analysis showed that air_pressure (28.51%) and rainfall (66.58%) exceed the 20% missing-value threshold, while AQI (4.06%), AQI_Bucket (4.06%), min_temp (2.87%), max_temp (6.09%), and wind_speed (11.48%) contain missing values below the threshold.

 rainfall column contains more than two-thirds of its values as missing. Although rainfall is an important meteorological factor that may influence air quality, such a high proportion of missing values could reduce data reliability  model training. Therefore, the rainfall column was removed from the dataset.
air_pressure column also exceeds the 20% threshold.percentage of missing values is  lower and the feature is considered an important meteorological variable that may contribute to AQI prediction. Removing the column would result in the loss of  information.the missing values in air_pressure were imputed using the median to preserve the feature for subsequent analysi remaining numeric feature columns (min_temp, max_temp, and wind_speed), missing values were also imputed using the **median**. The median was selected instead of the mean because it is more robust to outliers and skewed distributions.  mean, can be influenced by extreme values, the median provides a more representative measure of central tendency and helps produce more reliable imputations.

AQI and AQI_Bucket columns represent the target variables for the regression and classification models, respectively. Since supervised machine learning requires valid target labels for training, records containing missing values in either of these target columns cannot be used for model development. Instead of imputing artificial target values, all rows with missing AQI or AQI_Bucket values were removed from the dataset to ensure the integrity of the training data.

This preprocessing strategy balances data quality and feature retention while ensuring that the cleaned dataset is suitable for subsequent exploratory analysis and predictive model development.

Task 3 — Duplicate Analysis:

Duplicate records were checked using duplicated().

No duplicate records were found in the dataset; therefore, no rows were removed and the percentage of missing values remained unchanged.

Task 4 — Data Type Conversion

The following data type corrections were performed:

Converted the Date column to datetime.
Converted the Season column to the category data type to reduce memory usage.

Memory usage was compared before and after conversion.

Task 5 — Descriptive Statistics & Skewness

Descriptive statistics were generated for all numeric columns.
The highest absolute skewness was observed in AQI.

Positive skew indicates that a small number of unusually high AQI values pull the distribution toward the right, causing the mean to become larger than the median.

Negative skew indicates that a few unusually low values pull the distribution toward the left.

Since skewed data affects the mean more than the median, the median is generally preferred for imputing missing values. Median is the safer default whenever |skew| > 0.5, which covers almost every column in your table except max_temp.

Task 6 — Outlier Detection (IQR)

Analysis was performed on:

***AQI***
***Minimum Temperature***

The AQI column contained a larger proportion of outliers, while Minimum Temperature contained comparatively fewer.
outliers were not removed because extremely high AQI values may represent genuine pollution events rather than data entry errors. Removing them could reduce the model's ability to learn real-world pollution pattern

Outliers in AQI case is valid in this dataset since the datast used environemnetal factors, outliers will be retained during Part 1. Their impact on model performance will be evaluated during Part 2, and if required, suitable techniques such as capping or robust preprocessing methods will be considered based on the chosen machine learning model


Task 7 — Data Visualization

**In the code it is speacifcallyused output folder accroding to my project structure , if running this step casues any error kindly refer the outputs folder / juts comment this line and run all the code** 
The following visualizations were created:

Monthly AQI Trend (Line Plot)
Average AQI by Season (Bar Chart)
AQI Distribution (Histogram)
Wind Speed vs AQI (Scatter Plot)
AQI Distribution by Season (Box Plot)

#Histogram Interpretation


The histogram of the AQI column shows a positively (right) skewed distribution. Most AQI values are concentrated in the lower range (approximately between 50 and 200), a few rows have very high AQI values which is skewed right
the median is a more representative measure of central tendency than the mean for handling missing values.
The histogram also confirms the skewness value calculated earlier 4.05 ndicating a strong positive skew.

#Scatter Plot Interpretation

The scatter plot shows a weak negative relationship between wind speed and AQI. As wind speed increases, AQI generally tends to decrease, suggesting that stronger winds help disperse air pollutants.

#Box Plot Interpretation

The box plot shows noticeable differences in AQI across seasons. Winter and Post-monsoon exhibit higher median AQI values and greater variability than Summer and Monsoon, indicating seasonal differences in air pollution.

Task 8 — Correlation Analysis

 strong correlation between avg_temp and min_temp is expected  change together mainly due to seasonal weather conditions
Apart from the temperature columns, most features have weak correlations with AQI. For example, wind_speed has a weak negative correlation (-0.27) with AQI, suggesting that higher wind speeds may help disperse pollutants and reduce AQI. This indicates that AQI is affected by multiple climatic and environmental factors rather than a single variable.

Task 9(a) — Imputation Strategy Comparison
AQI mean: 155.24
AQI Median: 105.0
Min Temp Mean: 21.01
Min Temp Median: 22.73
Median was selected because:

AQI is positively skewed and the mean is influenced by extremely high values.
Minimum Temperature is negatively skewed and the mean is influenced by extremely low values.

Median therefore provides a more representative estimate for skewed data.After applying the chosen imputation strategy, all remaining missing values in the selected columns were handled successfullyut for AQI field since it will be my Target for my model null values are removed as in Part 2 specifically mentioned Supervised Learning , so actual Tragt value is required to train the data any imputation might affect the quality of model

Task 9(b) — Spearman Correlation

Spearman correlation was compared with Pearson correlation.

The largest differences were observed for:

Top 3 Pairs 
1. AQI-ONI  in Pearson it is 0.07 and in Spearman it is 0.27 and |DIfference| is 0.20 
2. min_temp-ONI   in Pearson it is -0.20 and in Spearman it is 0.36 and |DIfference| is 0.16
3. air_pressure-ONI  in Pearson it is 0.29 and in Spearman it is 0.44 and |DIfference| is 0.15 

For all three pairs, Spearman correlation was higher than Pearson correlation, indicating monotonic consistently goes one direction but non-linear(st linee) relationships.

For feature selection in Part 2, Pearson correlation will be used as the primary reference for linear relationships, while Spearman correlation will be used as supporting evidence for variables such as ONI where non-linear relationships exist.

#Task 9(c) — Grouped Aggregation

Season was selected as the categorical feature and AQI as the numeric feature.

Results showed:

Highest Mean AQI: Post-monsoon
Highest Standard Deviation: Post-monsoon

The high within-group standard deviation indicates that season alone is not sufficient to accurately predict AQI.

The ratio between the highest and lowest seasonal mean AQI was approximately 1.82, suggesting that season carries useful predictive information and should be included along with other climatic variables.

#Task 10 Cleanded CSV:
Cleaned Csv file is stored in required Data folder which will be used in next part 

#Output Files
Cleaned dataset (cleaned_data.csv)
Line Plot
Bar Chart
Histogram
Scatter Plot
Box Plot
Pearson Correlation Heat Map
Spearman Correlation Heat Map

