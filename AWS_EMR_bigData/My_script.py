from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType,FloatType

# Spark MLlib
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, LogisticRegression,RandomForestClassifier,GBTClassifier,\
                                      LinearSVC, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Pandas, numpy, sklearn ...
import pandas as pd
import numpy as np
import datetime
from time import time
from sklearn.feature_selection import SelectKBest, chi2
# import awswrangler as wr

# mini and full sparkify data set path
mini_sparkify_data_path = 's3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json'
full_sparkify_data_path = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"

sparkify_data_path = full_sparkify_data_path

my_s3_backet = "s3n://my-sparkify-data/my-data/"

createdFeatures_path = my_s3_backet+"data_model_big"


def get_time():
    '''return current time'''
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return time_now

################################################################################
#                              CLEAN DATA
################################################################################        

def clean_data(df):
    # 1. Drop any missing user_id or session_id
    df_clean = df.dropna(how='any', subset=['userId', 'sessionId'])

    # 2. Drop rows containing users with missing attributes (firstname, lastname...)
    df_clean = df_clean.filter(df_clean.firstName.isNotNull())

    # 3. Drop redundant columns
    df_clean = df_clean.drop('firstName')
    df_clean = df_clean.drop('lastName')

    # New dataset size
    print(f'Number of rows in the cleaned dataset: {df_clean.count()}')
    print(f'Number of columns in the cleaned dataset: {len(df_clean.columns)}')

    return df_clean


################################################################################
#                       CREATE, SAVE and LOAD FEATURES
################################################################################ 

def usage_stats(df_stats,feature,udf_function,column_udf,window, type_="ratio"):
    '''
    Return usage ratio or count of feature's catgories, where ratio =count of category /sum(count of all catgories).
    
    INPUT:
        df_stats: Pyspark dataframe
        feature: It is the column used to get usage ratio by feature.
                This feature is created or updated using udf_function.
                Example of feature: extract 'brower' from 'user_agent'.
        udf_function: The function applied to 'column_udf' to create or update the column 'feature' 
        column_udf: udf_function is applied to this column
        window: a window is applied.
        type_ (str): either ratio or count, default = ratio.
        
    OUTPUT:
        stats (Pyspark dataframe): a dataframe contating the usage ratio or count of feature's catgories
    '''

    stats = None
    if (type_=="ratio"):
        stats = df_stats.withColumn(feature, udf_function(df_stats[column_udf]))\
                   .select(["userId", feature]).groupBy(["userId", feature]).count()\
                   .withColumn("total", f.sum(f.col("count")).over(window))\
                   .withColumn("usage", f.col("count")/f.col("total")) \
                   .select(["userId", feature,"usage"])\
                   .groupBy(["userId"])\
                   .pivot(feature).sum()\
                   .orderBy("userId")\
                   .fillna(0)
    
    else: #count
        stats = df_stats.withColumn(feature, udf_function(df_stats[column_udf]))\
                   .select(["userId", feature]).groupBy(["userId", feature]).count()\
                   .groupBy(["userId"])\
                   .pivot(feature).sum()\
                   .orderBy("userId")\
                   .fillna(0)
        
    
    return stats    


def create_features(df_clean,s3_data_path):
    '''
    Create a dataframe with features and save the dataframe to json file.
    INPUT:
        df_clean (PySpark DF): the cleaned PySpark DF.
        path_to_json (str): Path to the JSON file where the created DF will be stored.
    
    OUTPUT:
        None
    '''    
    # 1. LABEL - CHURN    
    ###############################################################
    print(f"{get_time()} - get list of churners ...")
        
    cancel_event = udf(lambda x : 1 if x == 'Cancellation Confirmation' else 0, IntegerType())

    churners = df_clean.withColumn("churn", cancel_event("page"))\
    .select(['userId', 'churn'])\
    .groupBy('userId').agg(f.max('churn').alias("churn"))
    
    # 2. USER-INFO
    ###############################################################
    
    # 2.1 Create a dataFrame with three columns: userId, gender and level (precisely the latest level)
    print(f"{get_time()} - user info ...")
    
    user_info = df_clean.select(['userId', 'level','gender','ts'])\
    .orderBy(f.desc('ts'))\
    .dropDuplicates(['userId'])\
    .select(['userId', 'gender', 'level'])

    # 2.2 Convert Gender and level columns to numeric
    user_info = user_info.replace(['M', 'F'], ['0', '1'], 'gender')\
            .replace(['free', 'paid'], ['0', '1'], 'level')\
            .select('userId',f.col('gender').cast('int'), f.col('level').cast('int') )

    # 3. User_Agent: device and browser
    ###############################################################        
    
    window = Window.partitionBy("userId").rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
    
    print(f"{get_time()} - devices ...")
    get_device = udf(lambda x: x.split('(')[1].replace(";", " ").split(" ")[0])
    devices = usage_stats(df_clean,"device",get_device,"userAgent",window,type_="ratio")
    
    print(f"{get_time()} - browsers ...")
    get_browser = udf(lambda x: x.split(" ")[-1].split("/")[0])
    browsers = usage_stats(df_clean,"browser",get_browser,"userAgent",window,type_="ratio")

    # 4. Numerical features
    ###############################################################
    
    # 4.1 Overall stats (statistics for the whole period)
    print(f"{get_time()} - overall stats ...")
    overall_stats = df_clean.withColumn('state', f.split(df_clean['location'], ',')[1])\
               .select(["userId", "sessionId","artist","song","page","length","ts","registration","state"])\
               .groupby(["userId"])\
               .agg(f.count("userId").alias("count_logs"),\
                    f.countDistinct('sessionId').alias('count_sessions'),\
                    f.countDistinct('artist').alias('count_distinct_artists'),\
                    f.count('song').alias('count_songs'),\
                    f.countDistinct('song').alias('count_distinct_songs'),\
                    (f.sum('length')/(1000*60)).alias('sum_length_minutes'),\
                    ((f.max('ts') - f.max('registration'))/(1000*60*60*24)).alias("days_since_registraion")
                   )
    
    # 4.2 Stats per Sessions
    print(f"{get_time()} - stats per session ...")

    stats_per_session = df_clean.select(["userId", "sessionId","ts","song"])\
            .groupby(["userId","sessionId"])\
            .agg(((f.max("ts") - f.min("ts"))/(1000*60)).alias('minutes_per_session'),\
                 f.count("song").alias("count_songs_per_session"))\
            .groupby(["userId"])\
            .agg(
                 f.mean("count_songs_per_session").alias("avg_songs_per_session"),\
                 f.mean("minutes_per_session").alias("avg_session_length"),\
                 (f.sum("minutes_per_session")/60).alias("total_hours"), 
                )\
            .fillna(0)    
    
    # 5. Stats per page
    ###############################################################
    # window = Window.partitionBy("userId").rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
    
    pages_to_exclude = ['Cancel', 'Cancellation Confirmation', 'NextSong']
    
    # 5.1 Overall Stats per page (whole period)   
    print(f"{get_time()} - stats per page (count)...")
    
    rename_pages = udf(lambda x: "page_"+x.replace(" ", "_").lower())
    stats_per_page = usage_stats(df_clean.filter(~df_clean['page'].isin(pages_to_exclude)),
                                   "page",rename_pages,"page",window,type_="count")
    
    # 5.2 Hourly stats per page
    print(f"{get_time()} - hourly stats per page ...")

    stats_pages_hourly = stats_per_page\
    .join(stats_per_session.select("userId","total_hours"), ['userId'])\
    .fillna(0)

    for column in stats_per_page.drop("userId").columns:
        stats_pages_hourly = stats_pages_hourly.withColumn(column,f.col(column)/f.col("total_hours"))
        stats_pages_hourly = stats_pages_hourly.withColumnRenamed(column, column+"_hourly")

    stats_pages_hourly = stats_pages_hourly.drop("total_hours")
       
    # 6. Session interval
    ###############################################################
    print(f"{get_time()} - session interval ...")
    
    session_interval = df_clean \
        .groupby('userId','sessionId') \
        .agg(f.min('ts').alias('start_time'), f.max('ts').alias('end_time')) \
        .groupby('userId') \
        .agg(f.count('userId').alias('count_sessions'), \
            ((f.max('end_time') - f.min('start_time'))/(1000*60)).alias('observation_time'), \
            (f.sum(f.col('end_time') - f.col('start_time'))/(1000*60)).alias('total_sessions_time')) \
        .where(f.col('count_sessions') > 1)

    get_session_interval = udf(lambda x,y,z: (x-y)/(z-1), FloatType()) 

    session_interval = session_interval.withColumn("session_interval",\
                       get_session_interval(f.col('observation_time'),f.col('total_sessions_time'),f.col('count_sessions')))
    session_interval = session_interval.select("userId","session_interval")
    session_interval = session_interval.join(churners.select('userId'), 'userId', 'outer').fillna(0)
    
    # 7. Join all features together
    ###############################################################
    print(f"{get_time()} - Join all features together ...")

    model_df = churners\
            .join(user_info, ['userId'], 'outer')\
            .join(devices, ['userId'], 'outer')\
            .join(browsers, ['userId'], 'outer')\
            .join(overall_stats, ['userId'], 'outer')\
            .join(stats_per_session, ['userId'], 'outer')\
            .join(stats_pages_hourly, ['userId'], 'outer')\
            .join(session_interval, ['userId'], 'outer')\
            .fillna(0)   
    print('Count of users',':',model_df.count())       
    
    # 8. Save model_df to csv 
    ###############################################################
    print(f"{get_time()} - Save created features to parquet file ...")
    
    try:
        model_df.write.mode("overwrite").parquet(s3_data_path)
        print(f"Successfully saved to parquet file under {s3_data_path}!") 
    except:
        printt("Error saving file to s3 bucket!")
    

def load_features(spark,s3_data_path):
    '''
    Load created features.
    INPUT:
        spark: the spark session
        s3_data_path(str): the s3 path to json file.
    OUTPUT:
        model_df (PySpark DF): DF containg one row per user.
    '''
    model_df = spark.read.format("parquet").option("header","true").load(s3_data_path)

    # reorder columns. The first two columns must be "userId" and "churn". 
    cols = model_df.columns
    for col in ['userId','churn']:
        try:
            cols.remove(col)
        except:
            pass
    cols = ['userId','churn'] + cols
    model_df = model_df.select(cols)

    print("Data model (one row per user) reloaded...")
    print("Number of rows: ",model_df.count()) 
    print("Number of columns: ",len(cols),"\n") 
    
    return model_df


################################################################################
#                              FETAURE SELECTION
################################################################################

def remove_correltaed_features(model_pd):
    '''
    Remove highly correlated features using the `corr()` function from `Pandas`.
    INPUT:
        model_pd: The pandas DF containing our data model
    
    OUTPUT:
        features_to_keep (list): list of features to keep.
    '''
    dataset_pd = model_pd.drop(columns=['userId'])

    corr_matrix = dataset_pd.corr()
    correlated_columns = []
    for coln in corr_matrix.columns:
        correlated = corr_matrix.drop(coln, axis=0).loc[corr_matrix[coln].abs()>=0.5].index.tolist()
        if len(correlated) > 0:
            correlated_columns.append(coln)      
            
    # Find features to be removed, i.e. feature that has >0.8 correlation with any remaining feature
    corr = dataset_pd[correlated_columns].corr()
    columns_to_remove = []
    index_ = 0
    for col in corr.columns:
        index_ += 1
        if corr[col].iloc[index_:].max() >= 0.8:
            columns_to_remove.append(col)            
            
    print(f"{len(columns_to_remove)} highly correlated features that are removed:\n{columns_to_remove}\n\n")

    features_to_keep = dataset_pd.columns.drop(columns_to_remove).tolist()

    print(f"{len(features_to_keep)} features to keep including 'churn':\n{features_to_keep}")

    return features_to_keep

def feature_selection(data,num_best_features = 16):
    '''
    Selects features based on the k highest scores. The higher the score, the more important the feature is.
    INPUTS:
        num_best_features (int): number of best features to select, defualt =16.
        data (Spark DF): the Spark data frames containg the features
    
    OUTPUTS
        selected_best_features (list): List of the selected best features.
        featureScores (Pandas DF): DF containg the highest scores.
    '''
    for cols in data.columns.tolist():
        data = data[data[cols] >= 0] # To handle error "Input X must be non-negative"
    
    X = data.drop(columns=['churn'])  
    Y = data['churn']

    X_best= SelectKBest(chi2, k=num_best_features).fit(X, Y)

    # Creat Pandas DF with features and scores.
    featureScores = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(X_best.scores_)],axis=1)
    featureScores.columns = ['Feature','Score']  
    featureScores = featureScores.sort_values(by=['Score'],ascending=False)[:num_best_features].set_index("Feature")
    
    selected_best_features = list(featureScores.index)
    
    return selected_best_features,featureScores


################################################################################
#                              MODELING
################################################################################
def assign_class_weight(data):
    '''Assigh class weight to handle class imbalance'''
    y_collect = data.select('churn').groupBy('churn').count().collect()
    bin_counts = {y['churn']: y['count'] for y in y_collect}
    total = sum(bin_counts.values())
    n_labels = len(bin_counts)
    weights = {bin_: total/(n_labels*count) for bin_, count in bin_counts.items()}
    model_df = data.withColumn('weight', f.when(f.col('churn') == 1, weights[1]).otherwise(weights[0]))
    return model_df

def split_data(data, weights=[0.7,0.3], seed=42):
    """
    Splits data into training and testing subset, where .
    
    Args:
        data (DataFrame): The model data with features.
        seed (int): A seed value of the random number generator, default =42.
        weights (List[float]): split ratio, default: [0.7,0.3]
        
    Returns:
        train_df (DataFrame): The training subset.
        test_df (DataFrame): The testing subset.
    """
    train_df, test_df = data.randomSplit(weights, seed=seed);
    return train_df, test_df

def create_CrossValidator(classifier,data,isCV=True, paramGrid=None, numFolds=5,is_class_weight=False):
    '''
    Create a pipeline. If isCV is True, create a CrossValidator. 
    The pipeline includes normalizing and scaling numeric data and combining all features into feature vector series. 
    A classifier is the last stage in the pipeline.
    
    INPUTS:
        data (Spark DF): a dataframe with statistics per user.
        classifier: A machine learning classifier object. This is the last stage in the pipeline.
        isCV (boolean): whether or not a cross-validator is returned. If not, the pipeline is returned.
        paramGrid: a ParamGridBuilder object with hyperparameters.
        numFolds (int): number of folds of the CrossValidator, defualt=5.
        is_class_weight (boolean): whether there are class weights or not, default=False
        
    OUTPUT:
        crossval or pipeline: the CrossValidator or the pipeline, depending on the isCV parameter.
    '''
    # 1. Transform numerical features into a Vector structure  
   
    features = data.columns[2:-1] # the first two columns: 'userId' and 'churn'. The last column is:'weight'
        
    assembler = VectorAssembler(inputCols=features, outputCol="num_features") 

    # 2. Create a scaler object : MinMaxScaler
    scaler = MinMaxScaler(inputCol="num_features",outputCol="features") 
    # scaler = StandardScaler(withMean=True, withStd=True,inputCol="num_features",outputCol="features")
    
    # 3. Setup the pipeline
    pipeline  = Pipeline(stages = [assembler, scaler,classifier])    
    
    # 4. Cross validation
    crossval = pipeline  # ifCV = Flase return the pipeline
    
    if isCV:             
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=MulticlassClassificationEvaluator(labelCol="churn",metricName='f1'),
            numFolds=numFolds,
            seed=42,
        )        
    
    return crossval 

def train_model(classifier_name, crossval,data,print_msg=True):
    '''
    Train a cross validator.
    
    INPUT
        classifier_name: the name of the classifier
        crossval: the CrossValidator.
        data: the Spark DataFrame.
        print_msg (boolean): print message onto the screen.
    
    OUTPUT
        model: trained machine learning model
        training_time (float): the model's training time
    '''
    if print_msg:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Train {classifier_name} ...")
        

    # start = datetime.datetime.now()
    start = time()
    # fit the model to the data
    model = crossval.fit(data)
    
    # end = datetime.datetime.now()
    end = time()
    training_time = end - start
    
    return model, training_time

def evaluate_model(model,model_name,train,test,training_time,isCV=True,
                   paramGrid=None,best_params=None,num_best_features=16,seed=42,print_msg=True,num_params=None):    
    '''
    Evaluate model performance.
    
    INPUT
        model: trained machine learning model
        model_name: the model name
        train,test (Spark dataframe): the train and test dataframe.
        training_time (float): the model's training time
        isCV (boolean): whether a Cross Validation is used or not.
        paramGrid (list): parameter grid for hyperparameter tuning.
        best_params (dict): best hyperparameters of the CrossValidator.
        num_best_features (int): the number of features selected to feed the model.
        print_msg (boolean): print message onto the screen
        num_params (int): number of the parameters in the paramGrid
    
    OUTPUT
        evaluation_metrics (dict): disctionary of evaluation metrics, including:
                                accuracy, f1-score, precision, recall and AUC.
    '''
    # 1. make predictions
    predictions = model.transform(test)  
    predictions_train = model.transform(train)
    
    # 2. Instantiate MulticlassClassificationEvaluator and BinaryClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="churn",metricName="f1")
    evaluator_auc = BinaryClassificationEvaluator(labelCol='churn', rawPredictionCol='prediction', metricName='areaUnderROC')
    
    # 3. Evaluate test and train
    evaluation_metrics = {}
    evaluation_metrics["model"] = model_name
    evaluation_metrics["accuracy"] = evaluator.evaluate(predictions, {evaluator.metricName:"accuracy"})
    evaluation_metrics["accuracy (train)"] = evaluator.evaluate(predictions_train, {evaluator.metricName:"accuracy"})
    
    evaluation_metrics["f1_score"] = evaluator.evaluate(predictions)    
    evaluation_metrics["f1_score (train)"] = evaluator.evaluate(predictions_train)  
    
    if print_msg:
        print(f"  F1-score test ({model_name}) :{evaluation_metrics['f1_score']} ")
        print(f"  F1-score train ({model_name}) :{evaluation_metrics['f1_score (train)']} ")
    
    evaluation_metrics["precision"] = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    evaluation_metrics["recall"] = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    evaluation_metrics["auc"] = evaluator_auc.evaluate(predictions)
    evaluation_metrics["auc (train)"] = evaluator_auc.evaluate(predictions_train)
    
    evaluation_metrics["training_time"] = training_time
    evaluation_metrics["CrossValidation"] = isCV
    evaluation_metrics["paramGrid"] = paramGrid
    evaluation_metrics["num_params"] = num_params
    evaluation_metrics["best_params"] = best_params
    evaluation_metrics["num_best_features"] = num_best_features
    evaluation_metrics["seed"] = seed
    
    return evaluation_metrics  
# Get feature coefficinets

def get_feature_coefficinets(model,data, classifier_name):
    '''
    get feature coefficients of a model.
    
    INPUT:
        model: MLlib classifier
        data: Sparak dataframe
        classifier_name: classifier name
        
    Output:
        feature_coef_df (Pandas DF): a Pandas Dataframe with feature names and coefficients.        
    '''
     
    feature_cols = data.drop('userId','churn').columns
    
    if classifier_name=="LogisticRegression":
        feature_coef = model.coefficients.values.tolist()
        feature_name = feature_cols
    
    else:
        feature_coef = model.featureImportances.values.tolist()
        feature_ind = model.featureImportances.indices.tolist()
        feature_name = [feature_cols[ind] for ind in feature_ind]  
    
    feature_coef_df = pd.DataFrame(list(zip(feature_name, feature_coef)), columns=['Feature', 'Coefficient'])\
                    .sort_values('Coefficient', ascending=False)
          
    return feature_coef_df

def convert_paramGrid_to_list(paramGrid):
    '''
    convert ParamGrid structure to a simplified list.
    Example : paramGrid_gbt = ParamGridBuilder().addGrid(gbt.maxIter, [20, 40]).addGrid(gbt.maxDepth,[4, 5]).build()
              output : ['maxIter : [20, 40]', 'maxDepth : [4, 5]']
              
    INPUTS:
        paramGrid: the ParamGridBuilder
    
    OUTPUTS:
        params_list (list): list of parameters.
    
    '''    
    paramGrid_pd = pd.DataFrame([{p.name: v for p, v in m.items()} for m in paramGrid])
    params_list = []
    for col in paramGrid_pd:
        params_list +=[f"{col} : {list(paramGrid_pd[col].unique())}"]
        
    return params_list

def train_eval_withCV(model_df,model_pd,features_to_keep,paramGrid,classifier_list,
                      classifier_names,seed=42,isCV=True,num_Folds=5,num_best_features=10,
                      save_results=False,print_msg=True,is_class_weight=False,s3_backet=my_s3_backet):
    '''
    train and eval model using Cross Validation. `train_model` and `evaluate_model` are called.
    The steps are as follows:    
        1. Select K best features using SelectKBest, where k is the num_best_features(default = 10).
        2. train test split.
        3. Train and evaluate the models
        4. save results
        
    INPUTS:
        model_df (Spark DF): trained machine learning model
        model_pd (pandas DF): the equivalent Pandas DF
        features_to_keep: the feature to keep after dropping highly correlated features.
        paramGrid (dict): parameter grid for hyperparameter tuning.
        classifier_list (list): list of MLlib classifiers (LogisticRegression, RandomForestClassifier,...)
        classifier_names (list): list of MLlib classifier names
        seed (int): the seed used in `randomSplit`, default=42
        isCV (boolean): whether a Cross Validation is used or not.
        num_Folds (int): number of folds of the Cross validator.
        num_best_features (int): the number of features selected to feed the model.
        save_results (boolean): wether or not evaluation results are saved.
        print_msg (boolean): print message onto the screen
        is_class_weight (boolean): whether there are class weights or not.
        s3_backet : the s3 backet path.
        
    OUTPUTS:
        metrics_df (Pandas DF): DF containing evaluation results.
        models_list (list): list of created MLlib models
        selected_best_features (list): list of selected best features (using SelectKBest).
        model_topfeatures (Spark DF): DF containing the selected best features      
    '''

    # 1. Select best features
    selected_best_features,_ = feature_selection(model_pd[features_to_keep], num_best_features)
    
    model_topfeatures = model_df.select(["userId","churn"] + selected_best_features )  
        
    if is_class_weight:
        model_topfeatures = model_df.select(["userId","churn"] + selected_best_features + ["weight"])

    # 2. train test split
    train_df, test_df = split_data(model_topfeatures,seed=seed)
    # train_df, test_df = stratifiedSampler(model_topfeatures, ratio=0.7, label="churn", joinOn="userId",seed=seed)

    metrics_list = []
    models_list = []
    
    # 3. Train and evaluate the models

    for i in range(len(classifier_list)):
        crossval = create_CrossValidator(classifier_list[i],train_df,isCV,paramGrid[i],numFolds=num_Folds,
                                        is_class_weight = is_class_weight)

        # 3.1. train the model
        model,training_time = train_model(classifier_names[i],crossval,train_df,print_msg=print_msg)  

        # 3.2. Get best params (hig score)
        if isCV:
            scores = model.avgMetrics
            best_params = [{p.name: v for p, v in m.items()} for m in model.getEstimatorParamMaps()][np.argmax(scores)]
        else:
            best_params = None

        # 3.3. Evaluate the model
        params_list = convert_paramGrid_to_list(paramGrid[i])
        num_params = len(paramGrid[i]) # Get the number of parameters in the paramGrid (=length of all combinations)
        
        metrics = evaluate_model(model, classifier_names[i], train_df, test_df, training_time,
                                 isCV,params_list,best_params,num_best_features,seed=seed,
                                 print_msg=print_msg,num_params=num_params) 

        metrics_list.append(metrics)
        models_list.append(model)
    
    metrics_df = pd.DataFrame(metrics_list)

    # 4. save metrics
    if save_results:
        path = s3_backet+"metrics_CV_"+str(num_best_features)+"features_seed"+str(seed)+".csv"
        metrics_df.to_csv(path, index=False)
    
    return metrics_df, models_list,selected_best_features,model_topfeatures   


def run_experiment(model_df,model_pd,features_to_keep,isCV=True,num_Folds=5,
                            save_results=False,print_msg=False,is_class_weight=True):
    '''
    Build a series of K-fold cross validators using:
     - 1. N selected features where N = range(4,len(features_to_keep)-1,4).
     - 2. class weigh balancing
    '''
    # 10.1. initialize classifiers
    rfc = RandomForestClassifier(featuresCol="features",labelCol="churn",weightCol="weight",seed=42)
    lr = LogisticRegression(featuresCol="features",labelCol="churn",weightCol="weight")
    gbt = GBTClassifier(featuresCol="features",labelCol="churn",weightCol="weight",seed=42)
    svc = LinearSVC(featuresCol="features",labelCol="churn",weightCol="weight")
    dt = DecisionTreeClassifier(featuresCol="features",labelCol="churn",weightCol="weight",seed=42)

    classifier_list = [rfc,lr,gbt,svc,dt]
    classifier_names = ["RandomForestClassifier","LogisticRegression","GBTClassifier","LinearSVC","DecisionTreeClassifier"]
    
    # 10.2. Param Grid
    paramGrid = []
    for ii in range(len(classifier_list)): 
        paramGrid += [ParamGridBuilder().build()] 

    # 10.3. define seed_list and list_num_best_features
    seed_list = [42]
    list_num_best_features = range(4,len(features_to_keep)-1,4)
    
    # 10.4. initialise all_metrics_df : a dataframe that will concatenate all metrics.
    all_metrics_df = pd.DataFrame(columns=['model', 'accuracy', 'accuracy (train)', 'f1_score', 'f1_score (train)',
                                        'precision', 'recall', 'auc', 'auc (train)', 'training_time',
                                        'CrossValidation', 'paramGrid', 'best_params', 'num_best_features','seed'
                                        ])
    all_metrics_df['CrossValidation'] = all_metrics_df['CrossValidation'].astype('bool')
    all_metrics_df['seed'] = all_metrics_df['seed'].astype('int')

    for seed in seed_list:
        # for each num_best_features, generate a metrics DF, that will be concatenated with all_metrics_df
        for num_best_features in list_num_best_features:
            print(f"{get_time()}. Number of selected features = {num_best_features} and seed = {seed}. Please wait...")
            try:
                metrics_df, _,_,_ = train_eval_withCV(model_df,model_pd,features_to_keep,paramGrid,classifier_list,
                                                    classifier_names,seed=seed,isCV=isCV,num_Folds=num_Folds,
                                                    num_best_features=num_best_features,save_results=save_results,
                                                    print_msg=print_msg,is_class_weight=is_class_weight)

                all_metrics_df = pd.concat([all_metrics_df,metrics_df])
            except:
                pass
        
    all_metrics_df = all_metrics_df.reset_index().drop("index",axis=1)

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_metrics_df


################################################################################
#                              Main function
################################################################################    

def main():
    print(f"{get_time()}. Cleaning big data stored under ({sparkify_data_path}) and creating features...")

    # 1. Create spark session
    spark = SparkSession.builder.appName("Sparkify").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # 2. Load event logs:
    df = spark.read.format("json").option("header","true").load(sparkify_data_path)

    print(f'Row count: {df.count()}')
    print(f'Column count: {len(df.columns)}')
    df.printSchema()

    # 3. Clean dataset
    try:
        df_clean = clean_data(df)
        print("data cleaned successfully!")
    except:
        print("can not clean dataset!")

    # 4. Create features
    try:
        create_features(df_clean,s3_data_path=createdFeatures_path)        
    except:
        print("Can not create features!")

    # 5. Reload created features
    model_df = load_features(spark,s3_data_path=createdFeatures_path)

    # 6. Convert Pyspark DF to Pandas DF
    model_pd = model_df.toPandas()

    # 7. Remove highly correlted features
    features_to_keep =remove_correltaed_features(model_pd)

    # 8. feature_selection using SelectKBest
    # selected_best_features,featureScores = feature_selection(model_pd[features_to_keep],num_best_features = 10)
    # print("\nTop 10 features using 'SelectKBest' from 'sklearn':\n",featureScores)

    # 9. Assign class weights to handle class imbalance
    model_df = assign_class_weight(model_df)

    # 10. Modeling - Second experiment: same as second experiment, with class weighting
    print(f"\n{get_time()}. Train and eval classifiers vs. num selected features, with class weighting...\n")
    metrics_experiment = run_experiment(model_df,model_pd,features_to_keep,
                            isCV=True,num_Folds=5,save_results=False,print_msg=True,
                            is_class_weight=True)  

    metrics_experiment = metrics_experiment[['model', 'accuracy', 'accuracy (train)', 'f1_score', 
                        'f1_score (train)', 'precision', 'recall', 'auc', 'auc (train)', 
                        'training_time', 'seed','num_best_features']] 

    try:
        spark_metrics_DF = spark.createDataFrame(metrics_experiment) 
        spark_metrics_DF.write.mode("overwrite").parquet(my_s3_backet+"metrics_full_data")
    except:
        pass

    # 11. Feature importance of the GBTClassifier
    num_best_features = 8
    gbt = GBTClassifier(featuresCol="features",labelCol="churn",weightCol="weight",seed=42)
    classifier_list = [gbt]
    classifier_names = ["GBTClassifier"]

    paramGrid = [ParamGridBuilder().build()]
    metrics_df, models_list,selected_best_features,model_topfeatures=\
        train_eval_withCV(model_df,model_pd,features_to_keep,paramGrid,classifier_list,
                        classifier_names,seed=42,isCV=True,num_Folds=5,
                        num_best_features=num_best_features,save_results=False,
                        print_msg=True,is_class_weight=True)
    feature_coef_df = get_feature_coefficinets(models_list[0].bestModel.stages[-1],model_topfeatures,classifier_names[0]).round(2)
    feature_coef_df['Cumulative_coefficients'] = feature_coef_df['Coefficient'].cumsum()   
    print("\nFeature importance of GBT classifier ... full data ... EMR cluster:\n")
    print(feature_coef_df)
    


if __name__=="__main__":
    main()