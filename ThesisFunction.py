
"""
Thesis Utility Functions - UPDATED FOR REUTERS-ONLY DATASET

DEPRECATED FUNCTIONS (publisher category logic removed):
  - preprocess_news_categorization() - Use A_Reuters_DataPreprocessing.py
  - sentiment_analysis_on_categories_df() - Use B_FinBERT_Sentiment_Headlines.py
  - summarize_sa() - Use D_Consolidated_Data_Reuters.py

ACTIVE FUNCTIONS:
  - preprocess_news_date() - Associates weekend news to following Monday
  - sentiment_analysis_on_df() - FinBERT sentiment analysis on headlines
  - summarize_sa_topic() - Summarizes sentiment by topic
  - top_returns_correlations() - Gets features correlated with returns
  - df_normalization() - Normalizes numeric columns
  - my_confusion_matrix() - Calculates confusion matrix for classification
  - my_classification_scores() - Calculates classification metrics
  - get_significant_variables() - Logistic regression feature selection
  - get_final_significant_variables() - Iterative feature selection

NEW PIPELINE:
  1. A_Reuters_DataPreprocessing.py - Load and preprocess Reuters CSV data
  2. B_FinBERT_Sentiment_Headlines.py - Sentiment analysis
  3. C_BERTopic_Headlines.py - Topic modeling
  4. D_Consolidated_Data_Reuters.py - Merge all data
  5. E_Reuters_SP500_Correlation.py - Correlation analysis
  6. L_Pred_Models_Exploration.py - Model training
  7. M_Pred_Models_Evaluation.py - Model evaluation

@author: migue (updated for Reuters-only dataset)
"""


import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

import time
from datetime import datetime, timedelta


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

import statsmodels.api as sm

#-------------------------
#SAVE IN MULTIPLE PARQUETS
#-------------------------

def df_to_multiple_plarquets(df_to_slice,save_path, n_slices = 10):
    """Function that saves  dataframe in mulitple parquet files in the desired directory"""

    """
    Parameters
    ---------
    parquet_path: Desired path to save the parquet files
    n_slices: Number of parquet files to divide the dataframe
    ---------
    """

    slice_size = int(np.ceil(len(df_to_slice)/n_slices))

    #slicing for saving in multiple parquets
    for i in range(0, len(df_to_slice), slice_size):
        i_slice = int(i/slice_size)
        print("Saving slice: ",i_slice+1)
        df_slice = df_to_slice.iloc[i : i + slice_size]
        table = pa.Table.from_pandas(df_slice)
        pq.write_to_dataset(table,root_path= save_path)

    return("Parquets saved")



#-------------------------
#SAVE IN MULTIPLE PARQUETS
#   PARTITION BY YEAR
#-------------------------

def df_to_multiple_plarquets_partition(df_to_slice,save_path, n_slices = 10):
    """Function that saves  dataframe in mulitple parquet files in the desired directory (partition by year)"""

    """
    Parameters
    ---------
    parquet_path: Desired path to save the parquet files
    n_slices: Number of parquet files to divide the dataframe
    ---------
    """

    slice_size = int(np.ceil(len(df_to_slice)/n_slices))

    #slicing for saving in multiple parquets
    for i in range(0, len(df_to_slice), slice_size):
        i_slice = int(i/slice_size)
        print("Saving slice: ",i_slice+1)
        df_slice = df_to_slice.iloc[i : i + slice_size]
        table = pa.Table.from_pandas(df_slice)
        pq.write_to_dataset(table,root_path= save_path, partition_cols=["year"])

    return("Parquets saved")


#-------------------------
#-NEWS DATE PREPOCESSING--
#-------------------------

def preprocess_news_date(df_to_process):
    """Function that reads news from a dataframe and associates weekends news to fridays"""

    """
    Parameters
    ---------
    df_to_process: Data frame that is going to be processed
    ---------
    """
    news_df_sample = df_to_process.copy()
    news_df_sample = news_df_sample.sort_values("date")
    news_df_sample = news_df_sample.reset_index(drop=True)

    news_df_sample["dow"] = news_df_sample["date"].dt.dayofweek

    news_df_sample["weekday"] = news_df_sample["date"].dt.day_name()

    news_df_sample[news_df_sample["dow"]==6].head()

    #sunday and saturday associated to friday
    news_df_sample["associated_date"]=news_df_sample["date"]
    news_df_sample.loc[news_df_sample["dow"]==5, "associated_date"] = news_df_sample["date"] + timedelta(days=2)
    news_df_sample.loc[news_df_sample["dow"]==6, "associated_date"] = news_df_sample["date"] + timedelta(days=1)

    #news_df_sample.loc[news_df_sample["dow"]==5, "associated_date"] = news_df_sample["date"] - timedelta(days=1)
    #news_df_sample.loc[news_df_sample["dow"]==6, "associated_date"] = news_df_sample["date"] - timedelta(days=2)

    news_df_preprocessed = news_df_sample.copy()
    news_df_preprocessed = news_df_preprocessed.reset_index(drop=True)

    return(news_df_preprocessed)



#-------------------------
#-----DEPRECATED FUNCTION-
#-----NEWS CATEGORIES-----
#-------------------------
# DEPRECATED: Publisher categorization removed for financial news dataset
# Use A_Reuters_DataPreprocessing.py instead for data preprocessing
#
# def preprocess_news_categorization(df_to_process):
#     """DEPRECATED: Use A_Reuters_DataPreprocessing.py instead"""
#     pass

    #associated to friday
    #news_df_sample.loc[news_df_sample["dow"]==5, "associated_date"] = news_df_sample["date"] - timedelta(days=1)
    #news_df_sample.loc[news_df_sample["dow"]==6, "associated_date"] = news_df_sample["date"] - timedelta(days=2)

    news_df_preprocessed = news_df_sample.copy()
    news_df_preprocessed = news_df_preprocessed.reset_index(drop=True)

    return(news_df_preprocessed)







#-------------------------
#---SENTIMENT ANALYSIS----
#-------------------------

#Function to do sentiment analysis on the preprocessed data frame

#Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#Load Finbert Model
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def sentiment_analysis_on_df(df_to_analyze):
    """Function that reads news from a dataframe and process the sentiment analysis"""

    """
    Parameters
    ---------
    df_to_analyze: Data frame that is going to be processed with the sentiment analysis
    ---------
    """
    news_df_current = df_to_analyze.copy()

    titles_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    dates_list = []
    years_list =[]
    # get the start time
    st = time.time()


    for i in range(len(news_df_current)):
        text = news_df_current.iloc[i]['title']


        #Validate title is not empty
        if text != None:
            # Tokenize the text
            #print(i)
            input_id = tokenizer.encode(text, return_tensors="pt")
            my_tensor_size = input_id.size()[1]

            if my_tensor_size < 1000:
                output = finbert_model(input_id).logits

                predictions = torch.nn.functional.softmax(output, dim=-1).tolist()


                titles_list.append(news_df_current['title'].iloc[i])
                dates_list.append(news_df_current['associated_date'].iloc[i])
                years_list.append(news_df_current['year'].iloc[i])
                positive_list.append(predictions[0][0])
                negative_list.append(predictions[0][1])
                neutral_list.append(predictions[0][2])
                #if i > 2000:
                #    break


            else:
                print("skip news with large headline: ",i)


        #Print Process percentage
        percentage = round(((i)/len(news_df_current))*100,2)

        if (i == 1)|(i%1000 == 0):
            # get the current time
            ct = time.time()

            # get the execution time
            elapsed_time_seconds = round(ct - st,2)
            elapsed_time_minutes = round(elapsed_time_seconds/60,2)
            print('Processing Chunk: ', i, ' - percentage: ',str(percentage),'%',' (Elapsed time: ',elapsed_time_minutes,' minutes)')




    table = {'headline':titles_list,
            "date":dates_list,
            "year":years_list,
            "positive":positive_list,
            "negative":negative_list,
            "neutral":neutral_list
            }

    news_finbert_sa = pd.DataFrame(table, columns = ["headline", "date","year","positive", "negative", "neutral"])

    #news_finbert_sa.head()
    #calculating sentiment score
    news_finbert_sa["sa_score"]= news_finbert_sa["positive"] - news_finbert_sa["negative"]


    print('finished sentiment analysis')

    return(news_finbert_sa)






#-------------------------
#--DEPRECATED FUNCTION----
#--SENTIMENT ANALYSIS-----
#--WITH CATEGORIES--------
#-------------------------
# DEPRECATED: Category-based sentiment analysis removed
# Use B_FinBERT_Sentiment_Headlines.py instead for sentiment analysis
#
# This function is kept for reference but should not be used
# def sentiment_analysis_on_categories_df(df_to_analyze):
#     """DEPRECATED: Use B_FinBERT_Sentiment_Headlines.py instead"""
#     pass






#-------------------------
#--DEPRECATED FUNCTION----
#-SUMMARIZE SA BY------
#--CATEGORY (REMOVED)-----
#-------------------------
# DEPRECATED: Category-based summarization removed
# Use D_Consolidated_Data_Financial_News.py for data consolidation
#
# This function is kept for reference but should not be used
# def summarize_sa(df_to_summarize):
#     """DEPRECATED: Use D_Consolidated_Data_Reuters.py instead"""
#     pass



#-------------------------
#-SUMMARIZE SA and TOPIC--
#-------------------------

def summarize_sa_topic(df_to_summarize,n_dates_threshold = 78,min_prob = 0.5):
    """Function that reads a dataframe and summarize per day the mean score per topic"""

    """
    Parameters
    ---------
    df_to_summarize: Data frame that is going to be summarized
    n_dates_treshold: The function filters topcis to consider if they are above the 78 day treshold
    ---------
    """

    #Getting only docs with prob higher than 0.5
    documents_topics_df = df_to_summarize[(df_to_summarize["probs"] > min_prob)].reset_index(drop=True)
    #documents_topics_df = df_to_summarize.reset_index(drop=True)

    documents_topics_df["topic"] = "topic_" + documents_topics_df["topic"].apply(str)

    #Summarizing sa score per day per topic
    sa_topics_summary = documents_topics_df.groupby(["date","topic"])[["sa_score"]].mean().reset_index()

    date_count_per_topic = sa_topics_summary.groupby(["topic"])[["date"]].count().reset_index()
    date_count_per_topic = date_count_per_topic.sort_values(by="date",ascending=False)
    #date_count_per_topic
    #date_count_per_topic.describe()

    # Experiment Results
    # 25% 1st quartile is 78 days. 75% of the topics have 78 days or more.
    # max is 260 (total labor days) 78/ 260 -> 30%
    # at least 30% days of the year
    #day_count_threshold = 78

    day_count_threshold = n_dates_threshold
    #topcis to consider if they are above the 78 day treshold
    topics_to_consider = date_count_per_topic[(date_count_per_topic["date"] >= day_count_threshold)]['topic']

    sa_topics_to_consider_summary = sa_topics_summary[sa_topics_summary['topic'].isin(topics_to_consider)]

    #transpose
    sa_topic_summary = pd.pivot_table(sa_topics_to_consider_summary, values='sa_score', index=['date'], columns=['topic'], aggfunc="first").reset_index()

    return(sa_topic_summary)




#-------------------------
#-Returns Top Correlations--
#-------------------------

def top_returns_correlations(df_to_process,corr_treshold = 0.25):
    """Get correlation matrix with columns with a correlation larger than the determined threshold"""

    """
    Parameters
    ---------
    df_to_process: Data frame that is going to be processed
    correlation_treshold: the coefficient correlation defined as treshold
    ---------
    """

    initial_corr_matrix = df_to_process.corr()

    columns_to_drop = []
    columns_to_keep = []
    columns_to_keep_corr = []
    for column in initial_corr_matrix:
        returns_correlation = initial_corr_matrix.iloc[0][column]
        abs_returns_correlation = abs(initial_corr_matrix.iloc[0][column])

        if abs_returns_correlation >= corr_treshold:
            columns_to_keep.append(column)
            columns_to_keep_corr.append(returns_correlation)
        else:
            columns_to_drop.append(column)



    columns_to_keep_df = pd.DataFrame({"columns":columns_to_keep,
                                    "corr":columns_to_keep_corr
                                    })

    columns_to_keep_df =columns_to_keep_df.sort_values(by="corr",ascending=False)
    columns_to_keep_ordered = columns_to_keep_df['columns']


    corr_matrix = df_to_process[columns_to_keep_ordered].corr()

    return(corr_matrix)






#-------------------------
#-Variables Normalization-
#-------------------------

def df_normalization(df_to_process):
    """Add normalized columns to the df. Normaliztion on numeric columns"""

    """
    Parameters
    ---------
    df_to_process: Data frame that is going to be processed
    ---------
    """
    result_df = df_to_process.copy()

    #Normalize numeric columns
    for i in result_df.columns:
        numeric_flag = result_df[i].dtype == float
        column_name = i + '_z'
        if numeric_flag == True:
            result_df[column_name] = (result_df[i] - result_df[i].mean()) / result_df[i].std()

    return(result_df)







#----------------------------------
#-Classification Confusion Matrix--
#----------------------------------

def my_confusion_matrix(actual_df,predicted_l):
    """Function that calculates the confusion matrix"""

    """
    Parameters
    ---------
    actual_df: Data frame with the actual values
    predicted_l: List with the predicted values
    ---------
    """

    # True Positives
    tp = sum(actual_df[actual_df == predicted_l] == 1)
    # False Positive
    fp = sum(actual_df[actual_df != predicted_l] == 0)
    # False Negative
    fn = sum(actual_df[actual_df != predicted_l] == 1)
    # True Negatives
    tn = sum(actual_df[actual_df == predicted_l] == 0)

    return(tp,fp,fn,tn)





#-------------------------
#--Classification Scores--
#-------------------------

def my_classification_scores(tp,fp,fn,tn):
    """Function that scores the confusion matrix"""

    """
    Parameters
    ---------
    tp: True Positives
    fp: False Positives
    fn: False Negatives
    tn: True Negatives
    ---------
    """

    # Accuracy ACC
    # Sensitivity - True Positive Rate TPR - Recall
    # Specificity - True Negative Rate TNR
    # Precision - Positive Predicted Value PPV

    # calculated_accuracy_score = (tn+tp)/(tn+tp+fn+fp)
    # calculated_sensitivity_score = tp/(tp+fn)
    # calculated_specificity_score = tn/(tn+fp)
    # calculated_precision_score = tp/(tp+fp)

    calculated_accuracy_score = (tn+tp)/(tn+tp+fn+fp)
    calculated_sensitivity_score = tp/(tp+fn)
    calculated_specificity_score = tn/(tn+fp)

    # Cases in which the prediction are all negative. Division by zero
    if (tp == 0) and (fp == 0):
        calculated_precision_score = 0
    else:
        calculated_precision_score = tp/(tp+fp)


    results = [calculated_accuracy_score,calculated_sensitivity_score,calculated_specificity_score,calculated_precision_score]
    return(results)



#----------------------------------
#-GET SIGNIFICANT VARIABLES--------
#----------------------------------

def get_significant_variables(data,given_variables,pred_variable='direction'):
    """Function that runs a logistic regression and returns only the statistically significant variables"""

    """
    Parameters
    ---------
    data: Dataframe with the whole data set
    given_variables: Variables to consider
    pred_variable: name of the predicted variable
    ---------
    """

    if len(given_variables) > 0:
        X_values = data[given_variables]
        Y_values = data[pred_variable]
        log_regression = sm.Logit(Y_values, X_values).fit()

        significant_variables = log_regression.pvalues[log_regression.pvalues < 0.1].index.tolist()
    else:
        significant_variables =[]

    return(significant_variables)


def get_final_significant_variables(data,given_variables,pred_variable='direction'):
    """Function that runs logistic regressions until all variables are significant variables"""

    """
    Parameters
    ---------
    data: Dataframe with the whole data set
    given_variables: Variables to consider
    pred_variable: name of the predicted variable
    ---------
    """
    current_significant_variables = given_variables

    ncsv = 1
    nfsv = 2
    while ncsv != nfsv:
        current_significant_variables = get_significant_variables(data,current_significant_variables)
        ncsv = len(current_significant_variables)
        current_significant_variables = get_significant_variables(data,current_significant_variables)
        nfsv = len(current_significant_variables)

    final_significant_variables = current_significant_variables

    return(final_significant_variables)
