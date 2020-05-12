# Predicting Propaganda using BERT

<img src="toa-heftiba-QHuauUyXRt8-unsplash.jpg" width="150" height="200" />

<a href="https://medium.com/@bromas/predicting-propaganda-using-bert-bbb28b7deb71?source=friends_link&sk=92b472b78571058f86ab5e4fb679b35a">Medium article for project</a>


Guide to files:
* **EDA.ipynb**
  * Exploratory data analysis on training data. 
  * Leverage spaCy, intervaltree, textstat, and Matplotlib for visualizations.
  * Features an approach for increasing training data by 160%!
* **generate_data.ipynb**
  * Create training and dev .tsv files for classifiers.
  * Generate negative spans from news articles, increasing data by +160%.
* **dummy_classifier.ipynb**
  * Establish a baseline model for evaluating more sophisticated models.
  * Perform multi-class classification using training and dev data.
  * Evaluate performance using confusion matrix, classification report, and micro F1. 
* **logistic_regression.ipynb**
  * Create a logistic regression model.
  * Perform grid search to fine tune the model's hyperparameters.
  * Perform multi-class classification using training and dev data.
  * Evaluate performance using confusion matrix, classification report, and micro F1.
* **run_language_modeling.py**
  * Fine-tune a pre-trained BERT model on the training data. <span style="color:red"> This will require GPU and/or additional compute. Google Colab provides free GPU. </span>
  * This file comes from Hugging Face 🤗 and can be run as-is. <a href="https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py">Here is a link to the latest version on their repo.</a>
  * The call to run_language_modeling.py should look something like the following:
        python run_language_modeling.py \
          --model_type=distilbert \
          --model_name_or_path=distilbert-base-cased \
          --do_train \
          --train_data_file=datasets/train_data.tsv   \
          --output_dir=output \
          --mlm
  * Once the fine-tuning is completed there will be a number of files in the directory specified in output_dir, these comprise the new BERT model we'll use next.
* **bert.py**
  * Train and evaluate the BERT model.
  * This file currently only generates binary predictions.
  * File is meant to be run on a computing cluster. For instance, it was ran on U-M's Great Lakes cluster using SLURM.

Photo by Toa Heftiba on Unsplash (https://unsplash.com/photos/QHuauUyXRt8)
