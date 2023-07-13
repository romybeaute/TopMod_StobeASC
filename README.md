# Topic Modelling with LDA

This script performs topic modelling on the Dreamachine open report dataset using the Latent Dirichlet Allocation (LDA) method. The script includes data preprocessing, model training, model evaluation, and visualization of the results.

## Requirements

- Python 3.6 or later
- pandas
- os
- pathlib
- nltk
- re
- wordcloud
- gensim
- sklearn
- numpy
- matplotlib
- pyLDAvis

## Usage

1. Set the `condition`, `metaproject_name`, `subproject_name`, and `dataset_name` variables to match with different project and dataset.
2. Run the script. The script will read the dataset, preprocess the data, train an LDA model, evaluate the model, and visualize the results.

## Output

The script outputs the following:

- Information about the dataset
- A list of stop words used in the preprocessing step
- A word cloud visualization of the most common words in the dataset
- The number of reports in the dataset
- The best parameters for the LDA model found by grid search
- A heatmap of the grid search results
- Word cloud visualizations of the top 10 words for each topic
- The perplexity and coherence score of the LDA model
- An interactive visualization of the topics
- Word cloud visualizations of the most salient words for each topic
