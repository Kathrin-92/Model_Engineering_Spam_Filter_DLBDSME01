# A Case Study on the Development of a Spam Filter


## Table of Contents
1. [General Info](#General-Info)
2. [Setup](#Setup)


## General Info

### Project Overview

This repository contains the code and resources for a spam filter machine learning project, developed as part of the university course "Model Engineering". 
The project's main goal was to create, deploy, and assess machine learning models for spam detection using the CRISP-DM technique.

### Project Objective and Methods

This case study's goal was to assist a business that is putting in place a messaging/customer-facing system that needs a strong spam filter to help its customer support staff. 
Every stage of the CRISP-DM cycle is followed in this project.

* Machine Learning Models: Logistic Regression and Multinomial Naive Bayes (MNB)
* Deployment Tools: FastAPI for model serving, MLflow for model tracking and registry
* Project Approach: emphasis on iterative development, evaluation, and deployment readiness

### Results

* Two Trained Models: Logistic Regression and MNB, with MNB achieving better performance for this use case
* Detailed Evaluation Metrics: accuracy, precision, recall, and confusion matrix
* EDA Visualizations: graphs and charts showcasing key data insights and feature distributions

### Key Skills Learned

* Application of the CRISP-DM methodology in a real-world case study
* Data preprocessing and advanced exploratory data analysis techniques
* Development and evaluation of classification models using Python
* Model deployment with FastAPI and model tracking with MLflow


## Setup
### Prerequisites

For each project component, the requirements can be found in the respective requirements.txt files.

### Training Data

The training data was provided for by the university, but similar spam message traning data can be found online. 
Data needs to be in the form of a .csv file with two columns (message and label) for the code to work.
