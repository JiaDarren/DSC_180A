# Import Libraries and Data
import pandas as pd
import re

import numpy as np
from functools import reduce

from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

DATA_PATH = '../data'

# Feature Functions
def get_balance_stats(acct):
    """
    Balance Summary Stats: get balance summary statistics at the time of evaluation

    Params:
     - acct: Dataframe of account balances
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - minbalance: The maximum balance a consumer has in any of their accounts
        - maxbalance: The maximum balance a consumer has in any of their accounts
        - stdbalance: The standard deviation of all balances of a consumer's account
        - maxbalance_date: latest date of all features in dataset
    """
    balance_var = acct[['prism_consumer_id','balance','balance_date']].groupby('prism_consumer_id').agg({
        'balance':['min', 'max', 'std'],
        'balance_date':['max']
    }).reset_index()
    balance_var.fillna(0.0, inplace=True)

    balance_var.columns = balance_var.columns.to_flat_index().map(lambda x: x[1] + x[0])
    return balance_var

def get_disposable_income(total_cashflow):
    """
    Disposable Income: Gets disposable income a consumer has available after looking at
    all purchases and income sources in the dataset

    Params:
     - total_cashflow: Dataframe of all transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - total_balance: amount available after output and income transactions
        - total_balance_date: latest date of all features in dataset
    """
    total_balance = (
        total_cashflow[[
            'prism_consumer_id',
            'amount',
            'posted_date',
        ]]
        .groupby('prism_consumer_id')
        .agg({
            'amount':['sum'],
            'posted_date':['min','max']
        })
    )

    total_balance['date_range'] = pd.to_timedelta(total_balance['posted_date','max'] - total_balance['posted_date','min']).dt.days / 365
    total_balance = pd.DataFrame({
        'total_balance': total_balance['amount','sum'] / total_balance['date_range'], 
        'total_balance_date': total_balance['posted_date', 'max']
    }).reset_index()
    return total_balance

def get_monthly_balance_stats(outflows):
    """
    Monthly Balance Statistics: Features that look at average monthly activity

    Params:
     - outflows: Dataframe of outflow transactions and evaluation date
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - avg_monthly_spending: Average Monthly Spending
        - num_monthly_purchase: Average number of purchases made per month
    """
    outflow_counts = outflows.groupby('prism_consumer_id').agg({
        'amount':['count','sum'],
        'posted_date':['min','max'],
    }).reset_index()

    outflow_counts['date_range'] = pd.to_timedelta(outflow_counts['posted_date','max'] - outflow_counts['posted_date','min']).dt.days / 365 * 12

    outflow_counts = pd.DataFrame({
        'prism_consumer_id': outflow_counts['prism_consumer_id'],
        'avg_monthly_spending': outflow_counts['amount','sum'] / outflow_counts['date_range'],
        'num_monthly_purchase': outflow_counts['amount','count'] / outflow_counts['date_range'],
    })

    outflow_counts.replace([np.inf, -np.inf], 0.0, inplace=True)
    return outflow_counts

def get_num_savings_transfer(inflows):
    """
    Savings Feature: A count of how many times someone has pulled from savings account

    Params:
     - inflows: Dataframe with features
       ['prism_consumer_id', 'prism_account_id', 'memo_clean', 'amount', 
        'posted_date', 'category_description', 'evaluation_date']
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - num_savings_transfer: count of how many times someone has pulled from savings account
    """
    transfer_from_savings = inflows[inflows['category_description']=='SELF_TRANSFER']
    transfer_from_savings = transfer_from_savings[transfer_from_savings['memo_clean'].str.contains('Savings')]
    count_tfs = transfer_from_savings.groupby('prism_consumer_id').count().reset_index()
    inflow_ids = pd.merge(inflows[['prism_consumer_id']], count_tfs, on='prism_consumer_id', how='left')
    inflow_ids = inflow_ids.fillna(0).drop_duplicates(subset=['prism_consumer_id']).reset_index()[['prism_consumer_id', 'memo_clean']]
    inflow_ids.rename({'memo_clean':"num_savings_transfer"})
    
    return inflow_ids

def get_unsufficient_funds(acct):
    """
    Unsufficient Funds: Does a consumer have an account that is negative or near 0?

    Params:
     - acct: Dataframe of account balances
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - unsufficient_balance: boolean output for whether a consumer has an account that is negative or near 0.
    """
    acct = acct.copy()
    acct['unsufficient_balance'] = acct['balance'].apply(lambda x: x <= 1)
    unsufficient_accts = acct.groupby('prism_consumer_id')['unsufficient_balance'].count().reset_index()[['prism_consumer_id', 'unsufficient_balance']]
    return unsufficient_accts

def get_num_accounts(acct):
    """
    Category_Monthly_Slope: Calculate the monthly spending slope for each unique combination of 
    'category_description' and 'prism_consumer_id' in a dataset

    Params:
     - acct: Dataframe of account balances
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - account_count: total number of accounts each consumer has.
    """
    acct = acct.rename(columns = {'prism_consumer_id':'grouping_prism_consumer_id'})
    account_count = acct.groupby('grouping_prism_consumer_id')[['grouping_prism_consumer_id']].count()
    account_count = account_count.rename(columns = {'grouping_prism_consumer_id': 'account_count'}).reset_index()
    account_count = account_count.rename(columns = {'grouping_prism_consumer_id': 'prism_consumer_id'})
    return account_count

def get_cat_month_slp(outflows):
    """
    Category Monthly Slope: Calculate the monthly spending slope for each unique combination
    of 'category_description' and 'prism_consumer_id' in a dataset

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - [category name]_slp: slope of [category name]
    """
    # Group Data by Categories and Consumers
    cat_group = (
        outflows
        .groupby(['category_description', 'prism_consumer_id','posted_year', 'posted_month'])['amount']
        .sum()
        .reset_index()
    )
    
    # Calculate Slopes
    merged_df = pd.DataFrame()
    with np.errstate(invalid='ignore'):
        for i in cat_group['category_description'].unique():
            cat_name = str(i)
            id_5949 = cat_group[cat_group['category_description']== i]
            slopes = []
            for i in id_5949['prism_consumer_id'].unique():
                cat_id = id_5949[id_5949['prism_consumer_id']==i]
                slope, intercept, _, _, _ = linregress(range(len(cat_id)), cat_id['amount'])
                slopes.append(slope)
                
            df = pd.DataFrame({'prism_consumer_id': id_5949['prism_consumer_id'].unique(), cat_name + '_slp': slopes})
            
            if merged_df.empty:
                merged_df = df.copy()
            else:
                merged_df = pd.merge(merged_df, df, on='prism_consumer_id', how='outer')

    return merged_df

def get_non_essential_ratio(outflows):
    """
    Non-Essential Ratio: non-essential spending for each consumer based on defined essential categories

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - non_ess_ratio: ratio of spending spent on non-essentials
    """
    # a list of spending categories considered essential for each consumer.
    essential_lst = [
        'ACCOUNT_FEES',
        'GROCERIES','LOAN',
        'CREDIT_CARD_PAYMENT',
        'BILLS_UTILITIES',
        'AUTOMOTIVE',
        'INSURANCE',
        'ESSENTIAL_SERVICES',
        'HEALTHCARE_MEDICAL',
        'PETS','TAX','RENT', 
        'MORTGAGE', 'CHILD_DEPENDENTS'
    ]
    # Create a binary column indicating if the category is essential
    outflows = outflows.copy()
    outflows['essential'] = outflows['category_description'].isin(essential_lst).astype(int)
    
    # Calculate the non-essential ratio for each prism_consumer_id
    result = (
        outflows.groupby(['prism_consumer_id', 'essential'])['amount'].sum()
        .unstack(fill_value=0)
        .assign(non_ess_ratio=lambda x: x[0] / x.sum(axis=1))
        .reset_index()
    )
    
    id_lst = result['prism_consumer_id'].tolist()
    ess_ratio = result['non_ess_ratio'].tolist()
    df = pd.DataFrame({'prism_consumer_id': id_lst, 'non_ess_ratio': ess_ratio})
    
    return df

def get_stdzd_bal_slp(total_cashflow, acct):    
    """
    Standardized Balance Slope: Calculate the standardized balance slope of monthly balance

    Params:
     - total_cashflow: Dataframe of all transactions
     - acct: Dataframe of account balances
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - stdzd_bal_slp: ratio of spending spent on non-essentials
    """
    total_c = total_cashflow.copy()
    
    # Group by consumer, year, and month to sum the amounts
    total_c = total_c.groupby(['prism_consumer_id', 'posted_year', 'posted_month'])['amount'].sum().reset_index()
    
    # Get unique consumer IDs
    unique_ids = total_c['prism_consumer_id'].unique()
    
    # Initialize an empty list to store calculated slopes
    slopes = []
    
    # Loop through unique consumer IDs
    for i in unique_ids:
        # Filter data for the current consumer ID
        id_ = total_c[total_c['prism_consumer_id'] == i]
        
        # Get the initial balance for the consumer
        start_point = acct[acct['prism_consumer_id'] == i]['balance'].sum()
        
        # Reverse the monthly amounts and calculate cumulative balance
        diff_reversed = id_['amount'].iloc[::-1].tolist()
        line = np.cumsum(np.concatenate([[start_point], -np.array(diff_reversed)]))[::-1]
        
        # Use linregress to find the slope of the cumulative balance
        slope, intercept, _, _, _ = linregress(np.arange(len(line)), line)
        
        # Append the calculated slope to the list
        slopes.append(slope)
    
    # Create a DataFrame with consumer IDs and corresponding slopes
    df = pd.DataFrame({'prism_consumer_id': unique_ids, 'stdzd_bal_slp': slopes})
    
    # Standardize the slope values using StandardScaler
    features = df[['stdzd_bal_slp']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)    
  
    return pd.concat([df[['prism_consumer_id']], scaled_df], axis=1)

def get_PRR(total_cashflow):
    """
    Positive Remaining Ratio: Determines the amount of months where cashflow is 
    positive out of all months in the dataset for each consumer

    Params:
     - total_cashflow: Dataframe of all transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - PRR: ratio of positive cashflow monthly balances
    """
    # Group transactions by month and calculate remaining balance on each month for each client
    total_cashflow = total_cashflow.groupby(['prism_consumer_id', 'posted_year', 'posted_month'])['amount'].sum().reset_index()
    # Determine if the remaining amount is positive for each month
    total_cashflow['PRR'] = (total_cashflow['amount'] > 0).astype(int)
    # Calculate the ratio of months with positive remaining amounts
    total_cashflow = total_cashflow.groupby('prism_consumer_id')['PRR'].mean().reset_index()
    return total_cashflow

def get_CR(outflows):
    """
    Credit Ratio: Determines the maximum number of consecutive months 
    in which a consumer pays of a loan

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - CR: max number of consecutive months where consumer pays loan
    """
    # Get number of transactions for each category for each consumer
    cat_groups = (
        outflows
        .groupby(['prism_consumer_id', 'posted_date_id', 'category_description'])
        .size()
        .reset_index(name='count')
    )
    # Find number of months where consumer spent money on credit card payments or loans
    cat_counts = (
        cat_groups
        .groupby(['prism_consumer_id','category_description'])
        .agg({'count':'count', 'posted_date_id':['min', 'max']})
        .reset_index()
    )
    credit_counts = cat_counts[np.isin(cat_counts['category_description'], ['LOAN', 'CREDIT_CARD_PAYMENT'])].copy()
    # Get maximum credit ratio
    credit_counts['CR'] = (
        credit_counts['count','count'] / 
        (credit_counts['posted_date_id','max'] - credit_counts['posted_date_id','min'] + 1)
    )
    return credit_counts.groupby('prism_consumer_id')['CR'].max().reset_index()

def get_cat_stats(outflows):
    """
    Total Category Summary Statistics: summary statistics of category outflows

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - mean[category name]: mean of [category name]
        - median[category name]: median of [category name]
        - max[category name]: max of [category name]
        - min[category name]: min of [category name]
    """
    cat_aggs = (
        outflows
        .groupby(['prism_consumer_id', 'category_description'])['amount']
        .agg(['mean', 'median', 'min', 'max'])
        .reset_index()
        .pivot_table(
            index = 'prism_consumer_id',
            columns = 'category_description',
            values = ['mean', 'median', 'min', 'max']
        )
        .reset_index()
    )

    cat_aggs.columns = ["".join(a) for a in cat_aggs.columns.to_flat_index()]
    return cat_aggs

def avg_monthly_cat_num_trans(outflows):
    """
    Monthly Category Aggregates: get the average number of outflow transactions a consumer makes per month

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - [category name]_count: average number of [category name] outflow transactions per month 
    """
    # Get average number of transaction per month
    return (
        outflows
        .groupby(['prism_consumer_id', 'category_description', 'posted_date_id'])['amount']
        .count()
        .reset_index(name='count')
        .groupby(['prism_consumer_id', 'category_description'])
        .mean()
        .reset_index()
        .pivot(index='prism_consumer_id', columns='category_description', values='count')
        .add_suffix('_count')
        .reset_index()
        .rename_axis(None, axis=1)
    )

def avg_monthly_cat_spending(outflows):
    """
    Monthly Category Aggregates: get the average amount a consumer spends per month

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - [category name]_mean: average amount spent in [category name] per month 
    """
    # Get average amount of transaction per month
    return (
        outflows
        .groupby(['prism_consumer_id', 'category_description', 'posted_date_id'])['amount']
        .mean()
        .reset_index(name='mean_transaction')
        .groupby(['prism_consumer_id', 'category_description'])
        .mean()
        .reset_index()
        .pivot(index='prism_consumer_id', columns='category_description', values='mean_transaction')
        .add_suffix('_mean')
        .reset_index()
        .rename_axis(None, axis=1)
    )

def total_prop_spending(outflows):
    """
    Proportion of Spending: proportion of spending spent in each category

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - [category name]_prop: percentage amount spent in [category name]
    """
    cons_cat_spending = outflows.groupby(['prism_consumer_id', 'category_description'])['amount'].sum().reset_index()
    cons_total = outflows.groupby(['prism_consumer_id'])['amount'].sum().reset_index()

    cat_prop_indiv = cons_cat_spending.merge(cons_total, on='prism_consumer_id')
    cat_prop_indiv['prop'] = cat_prop_indiv['amount_x'] / cat_prop_indiv['amount_y']
    cat_prop_indiv = cat_prop_indiv[['prism_consumer_id', 'category_description', 'prop']]

    return (
        cat_prop_indiv.pivot(index='prism_consumer_id', columns='category_description', values='prop')
        .add_suffix('_prop')
        .reset_index()
        .rename_axis(None, axis=1)
        .fillna(0.0)
    )

def get_overdraft_freq(outflows):
    """
    Overdraft Frequency: Marks users that have more than 1 monthly overdraft transaction

    Params:
     - outflows: Dataframe of outflow transactions
    Returns:
     - Series with features:
        - index: prism_consumer_id (numerical ID of prism consumer)
        - values: int value indicating if user has more than 1 monthly overdraft transaction
    """
    outflows = outflows.copy()
    overdraft = outflows[outflows['category_description'] == 'OVERDRAFT']
    crosstab = pd.crosstab(overdraft['posted_date'], overdraft['prism_consumer_id'])

    outflow_count = pd.DataFrame(crosstab[crosstab > 1].stack()).reset_index()
    outflow_count = outflow_count.rename(columns = {'0': 'Overdraft Occurance'})
    outflow_count_index = pd.DataFrame(outflow_count.groupby('prism_consumer_id').count()['posted_date']).index

    outflows['Overdraft Count'] = outflows['prism_consumer_id'].isin(outflow_count_index)
    overdraft_users = outflows.groupby('prism_consumer_id')['Overdraft Count'].sum().apply(lambda x: 1 if x > 1 else 0)

    return overdraft_users


def add_transaction_ages(t_data, cons):
    """
    Helper function that adds date and age values to transaction dataset

    Params:
     - t_data: Dataframe of transaction data (inflow/outflow)
    Returns:
     - Transaction dataset with additional columns:
        - posted_month: month of transaction
        - posted_year: year of transaction
        - posted_date_id: id unique to month and year of posted date
        - eval_month: month of loan evaluation
        - eval_year: year of loan evaluation
        - eval_date_id: id unique to month and year of evaluation date
        - age: number of days between posted date and evaluation date
        - month_age: number of months between posted date and evaluation date
    """
    t_data = t_data.copy()
    t_data = t_data.merge(cons[['prism_consumer_id', 'evaluation_date']], on='prism_consumer_id', how='outer').dropna()
    
    t_data['posted_month'] = pd.to_datetime(t_data['posted_date']).dt.month
    t_data['posted_year'] = pd.to_datetime(t_data['posted_date']).dt.year
    t_data['posted_date_id'] = 12 * (t_data['posted_year'] - min(t_data['posted_year'])) + t_data['posted_month']

    t_data['eval_month'] = pd.to_datetime(t_data['evaluation_date']).dt.month
    t_data['eval_year'] = pd.to_datetime(t_data['evaluation_date']).dt.year
    t_data['eval_date_id'] = 12 * (t_data['eval_year'] - min(t_data['posted_year'])) + t_data['eval_month']

    t_data['age'] = (t_data['evaluation_date'] - t_data['posted_date']).apply(lambda x: x.days)
    t_data['month_age'] = t_data['eval_date_id'] - t_data['posted_date_id']
    return t_data

def get_cum_weighted_def_val(outflows, cons):
    """
    Cumulative Weighted Default Value: gets cumulative sum of default values weighted by age

    Params:
     - outflows: Dataframe of outflow transactions
     - cons: Dataframe of consumer evaluation times
    Returns:
     - Dataframe with features:
        - prism_consumer_id: numerical ID of prism consumer
        - cum_weighted_def_val: cumulative sum of default value
    """
    overdraft_outflows = (
        outflows[outflows['category_description'] == 'OVERDRAFT']
        [['prism_consumer_id', 'amount', 'age', 'month_age']]
    )

    overdraft_outflows['cum_weighted_def_val'] = (overdraft_outflows['amount'] / (1 + np.power(np.e, (overdraft_outflows['age'] / 30))))

    features_by_month = (
        overdraft_outflows.groupby(['prism_consumer_id','month_age'])
        .sum()
        .reset_index()
        .groupby('prism_consumer_id')[['cum_weighted_def_val']]
        .sum()
        .merge(cons[['prism_consumer_id']], on='prism_consumer_id', how='outer')
        .fillna(0.0)
    )
    return features_by_month

#==============#
# Main Process #
#==============#
def create_feature_matrix(cons, acct, inflows, outflows, output_file_path):
    # Create additional combination datasets
    print("Creating additional Datasets")
    # Create date values in outflows
    outflows = add_transaction_ages(outflows, cons)
    inflows = add_transaction_ages(inflows, cons)

    cash_outflow = outflows.copy()
    cash_inflow = inflows.copy()
    # add prefix to indicate inflow or outflow transaction category
    cash_inflow['category_description'] = 'inflow_' + cash_inflow['category_description']
    cash_outflow['category_description'] = 'outflow_' + cash_outflow['category_description']

    # make outflow transactions negative, and combine inflow and outflow datasets
    cash_outflow['amount'] = cash_outflow['amount'] * -1
    total_cashflow = pd.concat([cash_inflow, cash_outflow])
    total_cashflow = total_cashflow[total_cashflow['posted_date'] <= total_cashflow['evaluation_date']]
    
    # Get all Features
    print("Creating features")
    features = []
    features.append(get_balance_stats(acct))
    features.append(get_disposable_income(total_cashflow))
    features.append(get_monthly_balance_stats(outflows))
    features.append(get_num_savings_transfer(inflows))
    features.append(get_unsufficient_funds(acct))
    features.append(get_num_accounts(acct))
    features.append(get_cat_month_slp(outflows))
    features.append(get_non_essential_ratio(outflows))
    features.append(get_stdzd_bal_slp(total_cashflow, acct))
    features.append(get_PRR(total_cashflow))
    features.append(get_CR(outflows))
    features.append(get_cat_stats(outflows))
    features.append(avg_monthly_cat_num_trans(outflows))
    features.append(avg_monthly_cat_spending(outflows))
    features.append(total_prop_spending(outflows))
    features.append(get_overdraft_freq(outflows))
    features.append(get_cum_weighted_def_val(outflows, cons))

    # Merge all features into feature matrix
    print("Merging features to final feature matrix")
    feature_df = reduce(
        lambda left,right: pd.merge(left,right, on='prism_consumer_id', how='outer'), 
        features
    )

    feature_dates = re.findall(r"\w+_date", str(list(feature_df.columns)))
    feature_df['feature_date'] = feature_df[feature_dates].max(axis=1)
    feature_df.drop(feature_dates, axis=1, inplace=True)
    feature_df.fillna(0.0, inplace=True)

    # Final Feature Matrix
    sorted_cons = cons.sort_values('evaluation_date')

    dropped_cols = ['prism_consumer_id', 'evaluation_date', 'feature_date', 'APPROVED']
    feature_matrix = pd.merge(sorted_cons, feature_df, on='prism_consumer_id', how='left')

    # Make sure no invalid training data is pulled
    assert np.mean(feature_matrix['evaluation_date'] < feature_matrix['feature_date']) == 0, "Features pulled from dates after evaluation_date"
    feature_matrix.drop(dropped_cols, axis=1, inplace=True)

    # Remove problematic features
    problem_features = re.findall(
        r'(\w*(CHILD_DEPENDENTS|PETS)\w*)', 
        str(list(feature_matrix.columns))
    )

    feature_matrix.drop([x[0] for x in problem_features], axis=1, inplace=True)

    # Save final feature matrix
    print("Saving feature matrix")
    feature_matrix.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    cons = pd.read_parquet(f'{DATA_PATH}/raw/q2_consDF_final.pqt')
    acct = pd.read_parquet(f'{DATA_PATH}/raw/q2_acctDF_final.pqt')
    inflows = pd.read_parquet(f'{DATA_PATH}/raw/q2_inflows_final.pqt')
    outflows = pd.read_parquet(f'{DATA_PATH}/raw/q2_outflows_final.pqt')

    output_file_path = f"{DATA_PATH}/processed/test_feature_matrix.csv"

    create_feature_matrix(cons, acct, inflows, outflows, output_file_path)