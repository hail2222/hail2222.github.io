---
title:  "[Project] Adult Census Income"
excerpt: "Annual Income Analysis & Prediction"

categories:
  - projects
tags:
  - Blog, projects

toc: true

last_modified_at: 2024-09-03T08:06:00-05:00
---


***Annual Income Analysis & Prediction***

*A simple data analysis/ visualization/ prediction project, aimed at practicing handling big data using Python.*

***[See it in Github](https://github.com/hail2222/Adult-Census-Income)***

----------------

## 0. Introduction

Kaggle: Adult Census Income (<https://www.kaggle.com/datasets/uciml/adult-census-income/data>)

Briefly introducing the dataset, it is a data source generated based on the U.S. Census Bureau's data. It contains 14 features, including information such as age, education level, occupation, race, gender, and income. The goal of the project is to analyze relationships or patterns among features and create and evaluate an income prediction model.


## 1. Data Introduction
This dataset consists of a total of 48,842 instances, with 14 features and 1 target variable. The information for the 14 features is as shown below, and the target variable indicates whether the income is over or under $50,000.

The explanations for each feature are as follows:

- Age: Individual's age
  
- Workclass: Classification of individual's occupation (e.g., Public, Private, Unpaid)
  
- Education: Individual's education level (e.g., High School Graduate, Bachelor's degree, Master's degree)

- Marital Status: Individual's marital status (e.g., Married, Divorced, Separated)

- Occupation: Type of individual's occupation (e.g., Technical, Service, Management)

- Relationship: Individual's family relationship (e.g., Child, Parent, Husband, Wife)

- Race: Individual's race

- Sex: Individual's gender (Male or Female)

- Capital Gain: Capital gain obtained by the individual through investments or asset sales

- Capital Loss: Capital loss incurred by the individual through investments or asset sales

- Hours per week: Individual's weekly working hours

- Native Country: Individual's country of birth

- Income: Annual income of the individual, represented as "1" if above $50,000 and "0" otherwise. This is the target variable in the dataset.

Table below provides an overview of the overall structure of the dataset.
<img width="1097" alt="스크린샷 2024-01-25 오전 2 44 32" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/1e89f3d5-8f52-4d5a-bd4e-cb8af07ffce1">

## 2. Analysis Objectives

The main objective of this project is to understand various characteristics affecting adult annual income through data analysis and explore their correlations. We will investigate the relationship between each feature and annual income. If there are multiple features influencing annual income, we will consider the relationships between them through multivariate analysis. All analyzed aspects will be visualized through graphs during Exploratory Data Analysis (EDA).

Additionally, using machine learning models, we will predict annual income based on the given features and compare the accuracy of Logistic Regression and Support Vector Machine models.

## 3. Process

The data whole process is divided into EDA & Visualization and Machine-Learning.

### 3.1 EDA & Visualization

First, to examine the structure of the dataset, basic information for each column is checked. 
<img width="1028" alt="스크린샷 2024-01-25 오전 2 44 11" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/bb1dc670-b314-4267-9a55-0fdbca43970a">

Then, dataset is preprocessed by dropping data marked as '?'.
<img width="1097" alt="스크린샷 2024-01-25 오전 2 45 14" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/402cf9b2-432c-48a7-8e05-6417a70cd68e">


Next, EDA is conducted, and graphs for each column are generated. Since some columns, such as 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', contain numerical values, while others are categorical, separate graphs are created accordingly.

#### 1) Graphs for categorical columns:

- Workclass: Private dominates with the highest count.

- Education: Count is highest for High School Graduate, followed by College and Bachelors.

- Marital Status: Married-civ-spouse and never-married have the highest counts.

- Occupation: Distributed fairly evenly compared to other factors.

- Relationship: Highest counts for Husband and Not-in-family.

- Race: White is overwhelmingly predominant.

- Sex: Male count is about twice as high as Female.

- Native Country: United States nearly accounts for all samples.

- Income: Count is approximately 3 times higher for <=50K compared to >50K.

<img width="1093" alt="스크린샷 2024-01-25 오전 2 45 47" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/893acb80-c0a1-4e26-8aa5-699549dfa78d">


#### 2) Graphs for numerical columns:

- Age: Highest count in the 30s age group.

- fnlwgt: Shows an increase from 0 to 0.2, followed by a decrease from 0.2 onward.

- education.num: Corresponds to the education categories, similar to 'education'.

- Capital Gain, Capital Loss, Hours per week: Refer to the previous graphs.

<img width="1095" alt="스크린샷 2024-01-25 오전 2 46 21" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/da35b95f-de98-46b7-9f8f-af79e247d1b8">


#### 3) Various graphs:

Visualization including Multivariate graphs.

- Distribution of Income: Pie chart showing the count of individuals with income >=50K and <50K. <=50K has about three times more counts.

  <img width="1076" alt="스크린샷 2024-01-25 오전 2 46 44" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/96b8e0f4-074a-4fbd-b454-d8158b6548d2">

- Age distribution graph: The highest distribution is observed in the 30s age group. You can also see how the two categories(income >=50K and <50K) are distributed based on age.

  The age graph not only shows age distribution but also distinguishes between income <=50K (light blue) and >50K (dark blue). <=50K individuals are concentrated in the 20s to early 30s, while >50K individuals are widely distributed from late 30s to early 50s.


- Workclass: Violin plot is used to visualize the relationship between workclass and income. Private occupation has the widest distribution, indicating the highest income.
  In the workclass bar graph, individuals in both <=50K and >50K categories are predominantly in the Private occupation.

- Education: Individuals with income <=50K are mostly HS-grad, while those with income >50K are mostly Bachelors. Additionally, highest hours.per.week are seen on Prof-school, Doctorate, and Masters.


<img width="1093" alt="스크린샷 2024-01-25 오전 2 47 03" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/fd1f962e-d428-4a38-aa61-7ed88bf61d65">
<img width="1059" alt="스크린샷 2024-01-25 오전 2 47 31" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/536b1108-c096-4de1-b8dd-ae48024cf078">
<img width="1000" alt="스크린샷 2024-01-25 오전 2 48 02" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/db59d967-937d-401b-81cd-17019365cc2f">

### 3.2 Machine Learning (Predictions)

Using the analyzed data, machine learning models are employed to predict whether annual income is <=50K or >50K. 

#### 1) Data Processing

Data preprocessing involves excluding data with '?' in the 'occupation' column and converting the 'income' column values from string to numeric. '<=50K' is replaced with 0, and '>50K' is replaced with 1. Categorical values are converted to numerical values, and duplicate values are removed.

<img width="1094" alt="스크린샷 2024-01-25 오전 2 48 15" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/943f9dcf-42eb-4b37-8935-a8ec3156cba9">
<img width="1092" alt="스크린샷 2024-01-25 오전 2 48 27" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/d979929c-6d21-4beb-b29d-828d4c09853d">

#### 2) Training Models
Various models, including Logistic Regression, Random Forest, and Support Vector Machine, are compared for their accuracy in predicting annual income.

<img width="1095" alt="스크린샷 2024-01-25 오전 3 15 53" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/53a093d0-5d0d-4289-9366-3462b7243f8e">

#### 3) Classification report

Model accuracy and f1-score for each model are as follows (rounded to two decimal places):

<img width="1095" alt="스크린샷 2024-01-25 오전 3 16 18" src="https://github.com/hail2222/Adult-Census-Income/assets/100838589/27c55ce7-48f8-4640-a01a-c1aeead30666">

## 4. Analysis Results

The results can be broadly categorized into data analysis results and machine learning results.

### 4.1 Data Analysis Results

Summarizing the observed relationships of features from data analysis:

- <=50K individuals are about three times more numerous than >50K individuals.

- Private occupation dominates among workclass.

- High School Graduate, College, and Bachelors are the most common education levels.

- Married-civ-spouse and never-married individuals have the highest counts in marital status.

- Male count is about twice as high as Female count.

- United States accounts for almost all individuals in the native country.

- Highest age distribution is observed in the 30s.

- fnlwgt increases up to 0.2, then shows a decreasing trend.

- Private occupation has the highest income.

- Highest age distribution is observed in the 30s.

- Education X Hours.per.week: Prof-school, Doctorate, and Masters show the highest hours.per.week.

### 4.2 Machine Learning Results

Using the classification report, the performance of the three models is analyzed in detail. Looking at Precision and f1-score, Random Forest performs the best for both <=50K and >50K categories. Therefore, when predicting income with this dataset, using the Random Forest model would yield the most accurate results.

#### 1) Evaluation

##### 1. Precision:

	•	<=50K: 0.89
	•	>50K: 0.74

The precision for predicting individuals earning <=50K is 0.89, which is quite high, indicating that the model is generally accurate when predicting this class. However, the precision for predicting >50K is lower at 0.74, which suggests that the model has more false positives when predicting high-income earners.

##### 2. Recall:

	•	<=50K: 0.93
	•	>50K: 0.64

Recall for the <=50K group is 0.93, meaning the model captures most of the actual <=50K earners, which is a strong result. However, the recall for >50K is only 0.64, meaning that the model is missing a significant portion of individuals who actually earn more than 50K. This suggests a higher number of false negatives for the >50K class.

##### 3. F1-Score:

	•	<=50K: 0.91
	•	>50K: 0.69

The F1-Score, which balances precision and recall, is 0.91 for <=50K, showing that the model is performing very well for this class. For >50K, the F1-Score is 0.69, reflecting a weaker performance in predicting high-income earners.

##### 4. Support:

	•	<=50K: 6867
	•	>50K: 2175

There is a noticeable class imbalance, with nearly three times as many <=50K earners compared to >50K earners. This class imbalance likely contributes to the lower performance in predicting the >50K class.

Overall Evaluation:

	•	<=50K (Low Income): The model performs excellently for this class, with high precision (0.89), recall (0.93), and F1-Score (0.91). It effectively predicts low-income individuals.
	•	>50K (High Income): The model struggles more with high-income individuals, as seen from the lower precision (0.74), recall (0.64), and F1-Score (0.69). The model is missing many true high-income earners, and its predictions for this class are less accurate.
	•	Class Imbalance: The support values indicate class imbalance, which is likely a significant factor in the performance disparity between the two classes.

Suggestions for Improvement:

	1.	Address Class Imbalance: Methods like SMOTE, oversampling, or undersampling could help balance the data and improve the model’s performance for the >50K class.
	2.	Use Different Metrics: Evaluating the model using additional metrics like ROC-AUC could provide better insights, especially given the imbalance.
	3.	Tuning the Model: Adjusting the class weights or fine-tuning the decision threshold might help improve the prediction of high-income individuals.

Currently, the model performs well for the <=50K class, but improvement is needed for the >50K class, especially in terms of recall.






Title of this post is {{ page.title }},
and the last modified time is {{ page.last_modified_at }}.