# DSC180 Project Git Repository
## Getting Started:
To build the environment, run: 'pip install -r requirements.txt'

To see current progress, run 'risk_predictor.ipynb' in the notebooks folder.

## Project Objective:
Build a score to predict a given consumer's default risk based on their banking transaction history

## Current Progress:
During Quarter 1, we built several models to categorize inflow and outflow transactions based on the transaction memo. From our results, we ended up using a SGD model using tfidf features, which had an accuracy of around 97%. 

During Quarter 2, using those categorizations, the date of each transaction, and some other data provided in the datasets, we attempt to build a model to predict whether a consumer will default on their first loan payment, given their previous transaction history.

## Running the Model:
### Directory and Data Setup:
Because the data used in this project is protected, the data directory is not included. To keep the directory structure consistent, create a 'data' directory in the main git directory. Within the data directory, create the following empty directories: 'raw' and 'processed'.

To retrain/run any models for yourself, ensure you have access to the following datasets:

- **q2_consDF_final.pqt**: parquet file with the consumer id and date of loan evaluation
  - 'prism_consumer_id': unique id for each consumer
  - 'evaluation_date': date the consumer was evaluated for a loan
  - 'APPROVED': If a consumer was approved for a loan (this is 1 (true) for all consumers)
- **q2_acctDF_final.pqt**: parquet file with the account balances of each account for each consumer
  - 'prism_consumer_id': unique id for each consumer
  - 'prism_account_id': unique id for each consumer's account
  - 'balance': balance of consumer's account at balance date
  - 'balance_date': date consumer's account balance was checked
  - 'account_type': type of account (savings, checking, etc.)
- **q2_inflows_final.pqt**: parquet file with all the inflow transactions of every consumer
  - 'prism_consumer_id': unique id for each consumer
  - 'prism_account_id': unique id for each consumer's account
  - 'memo_clean': transaction description
  - 'amount': amount of transaction
  - 'posted_date': date transaction was posted
  - 'category_description': transaction category description
- **q2_outflows_final.pqt**: parquet file with all the outflow transactions of every consumer
  - 'prism_consumer_id': unique id for each consumer
  - 'prism_account_id': unique id for each consumer's account
  - 'memo_clean': transaction description
  - 'amount': amount of transaction
  - 'posted_date': date transaction was posted
  - 'category_description': transaction category description
*If you have access to the dsc180-fa23, clone the relevant github repo in the same directory as this repo and run dataset_init.ipynb to set up the raw datasets.*

### Feature Evaluation
**If you just want to run the model saved in the models directory, skip this section**
Once you have all the raw datasets, run 'create_features.py' to build the feature matrix. Make sure the paths and file names corresponse to the correct datasets.

Run 'eval_feature.py' to evaluate feature impact with feature permutation method.

Run 'reduce_features.py' to create feature matrix with reduced features.

### Build Model
**If you just want to run the model saved in the models directory, skip this section**
In the event that there isn't a model saved in the models directory, or you want to train your own model, run 'model_testing.ipynb'. Make sure that the file paths correspond to the appropriate datasets. 

### Model Prediction
To predict a dataset, run 'risk_predictor.ipynb'. Make sure the file paths are set to the appropriate datasets. The predictions will be saved at 'prediction_path'.