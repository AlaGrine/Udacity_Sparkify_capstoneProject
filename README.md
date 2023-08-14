# Sparkiy: churn prediction with Spark

## DSND Capstone Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#file_descriptions)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Acknowledgements](#Acknowledgements)

## Project Motivation <a name="motivation"></a>

This project is part of [Udacity](https://www.udacity.com/)'s Data Science Nanodegree Program.

The aim of this project is to build a binary classification model using [Pyspark ML](https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html) to predict churn for Sparkify.

Udacity provided a **12GB** dataset of customer activity from Sparkify, a fictional music streaming service similar to Spotify. The dataset logs user interactions with the service, like listening to streaming songs, adding songs to playlists, thumbs up and down, etc.

Tiny (125MB) and medium (237MB) subsets of the full dataset are also provided.

[PySpark](https://spark.apache.org/docs/latest/api/python/index.html), the Python API for Apache Spark, is used here on both the local machine and the AWS EMR cluster.

The project is divided into the following sections:

1. Use the **small subset** (on a local machine) to perform `exploratory data analysis` and build a `prototype machine learning model`.
2. Scale up: use the **medium dataset** (on a local machine) to see if our model works well on a larger dataset.<br>
3. Deploy a cluster in the cloud with [AWS](https://aws.amazon.com/console/) using the **full 12GB dataset**.

## Installation <a name="installation"></a>

This project requires Python 3, Spark 3.4.1, and the following Python libraries installed:

`Pyspark` ,`Pandas`, `Numpy`, `scipy`, `Plotly` and `Matplotlib`

## File Descriptions <a name="file_descriptions"></a>

**The main file of the project** is `Sparkify.ipynb`, which uses the small dataset and can therefore be run locally.

The project folder also contains the following:

- `Sparkify_medium.ipynb`: The medium dataset which you can run locally.
- `metrics` folder: The metrics of our models, including f1-score and training time, are available here (csv files).
- `statistics` folder: Descriptive features of the main characteristics of small and medium datasets are available here (csv files).
- `AWS_EMR_bigData` folder: It contains the inputs you can upload to your S3 bucket, and the outputs you can download from the same bucket.

  - `My_script.py`: The Python script to run on the EMR cluster. You will need to upload it to your S3 bucket first.
  - `install-my-jupyter-libraries`: A schell script you need to upload to your s3 bucket before creating your EMR cluster. When you create your cluster, add this script as a bootstrap action to install the required libraries.

  - `S3_download` folder: Contains the metrics of the model executed on the EMR cluster. downloaded from my s3 bucket.

## Instructions <a name="instructions"></a>

1. All you need to do is unzip the JSON files provided by Udacity to run the code on your local machine.

2. To run the Python script on the AWS EMR cluster, you need to submit the script to your cluster through the command line as follows:

   `aws s3 cp s3://your_backet_name/My_script.py .`

   `spark-submit My_script.py`

   The first command downloads the script from your S3 backet to the master machine for execution.

## Results<a name="results"></a>

I wrote a blog post about this project. You can find it [here](https://medium.com/@alaeddine.grine/sparkify-churn-prediction-with-pyspark-47f3c166a952).

## Acknowledgements<a name="Acknowledgements"></a>

Must give credit to [udacity](https://www.udacity.com/) for making this a wonderful learning experience.
