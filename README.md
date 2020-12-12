# About
Repository for CS 591 W1 Active Learning in Machine Translation and Error Classification.

# How to Run

## Error Classification of Machine Translation

### Generating Error Labels
1. Navigate to Experiments/Error Classification/Generate Error Labels For Translations.ipynb
2. Run the code under different sections marked in the file. Remember to change paths where applicable for the machine translation results and also for loading the pre-trained Error Classification model. 

### Collecting Error Classification Statistics
1. Navigate to Experiments/Error Classification/classification_stats.ipynb
2. Change paths where applicable for the main folder where the error labels are present (by default in the code, the error_predictions.csv file is stored under the same directory of the machine translation results under the specific layer/head for a particular budget).
3. Run the code under different sections marked in the file. 

### Plotting Error Classification Results
1. Navigate to Experiments/Error Classification/classification_plots.ipynb
2. Change paths where applicable. 
3. Run the code under different marked down sections in the file for specific plots. 

