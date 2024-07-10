# Credit Card Users Churn Prediction


## Problem Statement

### Business Context

The Thera bank recently saw a steep decline in the number of users of their credit card, credit cards are a good source of income for banks because of different kinds of fees charged by the banks like annual fees, balance transfer fees, and cash advance fees, late payment fees, foreign transaction fees, and others. Some fees are charged to every user irrespective of usage, while others are charged under specified circumstances.

Customers’ leaving credit cards services would lead bank to loss, so the bank wants to analyze the data of customers and identify the customers who will leave their credit card services and reason for same – so that bank could improve upon those areas

You as a Data scientist at Thera bank need to come up with a classification model that will help the bank improve its services so that customers do not renounce their credit cards

### Data Description

* CLIENTNUM: Client number. Unique identifier for the customer holding the account
* Attrition_Flag: Internal event (customer activity) variable - if the account is closed then "Attrited Customer" else "Existing Customer"
* Customer_Age: Age in Years
* Gender: Gender of the account holder
* Dependent_count: Number of dependents
* Education_Level: Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to college student), Post-Graduate, Doctorate
* Marital_Status: Marital Status of the account holder
* Income_Category: Annual Income Category of the account holder
* Card_Category: Type of Card
* Months_on_book: Period of relationship with the bank (in months)
* Total_Relationship_Count: Total no. of products held by the customer
* Months_Inactive_12_mon: No. of months inactive in the last 12 months
* Contacts_Count_12_mon: No. of Contacts in the last 12 months
* Credit_Limit: Credit Limit on the Credit Card
* Total_Revolving_Bal: Total Revolving Balance on the Credit Card
* Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)
* Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
* Total_Trans_Amt: Total Transaction Amount (Last 12 months)
* Total_Trans_Ct: Total Transaction Count (Last 12 months)
* Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
* Avg_Utilization_Ratio: Average Card Utilization Ratio

#### What Is a Revolving Balance?

- If we don't pay the balance of the revolving credit account in full every month, the unpaid portion carries over to the next month. That's called a revolving balance


##### What is the Average Open to buy?

- 'Open to Buy' means the amount left on your credit card to use. Now, this column represents the average of this value for the last 12 months.

##### What is the Average utilization Ratio?

- The Avg_Utilization_Ratio represents how much of the available credit the customer spent. This is useful for calculating credit scores.


##### Relation b/w Avg_Open_To_Buy, Credit_Limit and Avg_Utilization_Ratio:

- ( Avg_Open_To_Buy / Credit_Limit ) + Avg_Utilization_Ratio = 1





## Exploratory Data Analysis (EDA)


- EDA is an important part of any project involving data.
- It is important to investigate and understand the data better before building a model with it.
- A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
- A thorough analysis of the data, in addition to the questions mentioned below, should be done.


**Questions**:

1. How is the total transaction amount distributed?
2. What is the distribution of the level of education of customers?
3. What is the distribution of the level of income of customers?
4. How does the change in transaction amount between Q4 and Q1 (`total_ct_change_Q4_Q1`) vary by the customer's account status (`Attrition_Flag`)?
5. How does the number of months a customer was inactive in the last 12 months (`Months_Inactive_12_mon`) vary by the customer's account status (`Attrition_Flag`)?
6. What are the attributes that have a strong correlation with each other?





### Model evaluation criterion

The nature of predictions made by the classification model will translate as follows:

- True positives (TP) are failures correctly predicted by the model.
- False negatives (FN) are real failures in a generator where there is no detection by model.
- False positives (FP) are failure detections in a generator where there is no failure.

**Which metric to optimize?**

* We need to choose the metric which will ensure that the maximum number of generator failures are predicted correctly by the model.
* We would want Recall to be maximized as greater the Recall, the higher the chances of minimizing false negatives.
* We want to minimize false negatives because if a model predicts that a machine will have no failure when there will be a failure, it will increase the maintenance cost.



#### Model can make wrong predictions as:

* Predicting a customer will not renounce their credit cards but in reality the customer will (FN).
* Predicting a customer will renouuce their credit card but in reality the customer will not (FP).

#### Which case is more important?
* If we predict that a customer will not renounce their credit cards, but in reality the customer renounces it, this will lead the bank to loss of income.
* If we predict that a customer will renounce their credit cards, and they end up not renouncing it, this will lead to loss of resources.

#### How to reduce loses?
* `recall` should be maximized, the greater the recall, the higher the chances of minimizing false negatives.



## Model Comparison and Final Model Selection

I trained several models to compare performance before selecting the best model. See trained model list below:

GradientBoostingClassifier with original data
GradientBoostingClassifier with oversampling data 
GradientBoostingClassifier with undersampling data

AdaBoostClassifier with original data
AdaBoostClassifier with oversampling data
AdaBoostClassifier with undersampling data

DecisionTreeClassifier with original data
DecisionTreeClassifier with oversampling data
DecisionTreeClassifier with undersampling data

BaggingClassifier with original data
BaggingClassifier with oversampling data
BaggingClassifier with undersampling data

RandomForestClassifier with original data
RandomForestClassifier with oversampling data
RandomForestClassifier with undersampling data  

XGBClassifier with original data
XGBClassifier with oversampling data
XGBClassifier with undersampling data


Finally,
* XGBoost trained with oversampling data, XGBoost trained with undersampling data, and Bagging trained with undersampling data had a perfect recall score on validation data, and giving a gerneralized performance as compaered to other models.

* XGBoost trained with oversampling data model is the best model among the above mentioned model as it has the highest Accuracy and precision score when compare to XGBoost trained with undersampling data, and Bagging trained with undersampling data.


# Checking the model performance on test set:
	Accuracy	Recall	Precision	F1
0	0.64116	    1.00000	0.30894	0.47204

* The XGBoost model trained on oversampled data has given ~100% recall on the test set
* This performance is in line with what we achieved with this model on the train and validation sets
* So, this is a generalized model










# Business Insights and Conclusions

* We have been able to build a predictive model:

  a) that bank can deploy this model to identify customers who are at the risk of attrition.

  b) that the bank can use to find the key causes that drive attrition.

  c) based on which bank can take appropriate actions to build better retention policies for customers.


* Factors that drive the attrition - Total_Trans_Ct, Total_Trans_Amt, months_inactive_12_mon, Total_Relationship_Count, Total_Revolving_Bal
* Total_Trans_Ct: Less number of transactions in a year leads to attrition of a customer - to increase the usage of cards the bank can provide offers like cashback, special discounts on the purchase of something, etc so that customers feel motivated to use their cards.

* Total_Revolving_Bal: Customers with less total revolving balance are the ones who attrited, such customers must have cleared their dues and opted out of the credit card service. After the customer has cleared the dues bank can ask for feedback on their experience and get to the cause of attrition.

* Total_Trans_Amt: Less number of transactions can lead to less transaction amount and eventually leads to customer attrition - Bank can provide offers on the purchase of costlier items which in turn will benefit the customers and bank both.

* Total_Relationship_Count: Attrition is highest among the customers who are using 1 or 2 products offered by the bank - together they constitute ~55% of the attrition - Bank should investigate here to find the problems customers are facing with these products, customer support, or more transparency can help in retaining customers.

* Female customers should be the target customers for any kind of marketing campaign as they are the ones who utilize their credits, make more and higher amount transactions. But their credit limit is less so increasing the credit limit for such customers can profit the bank.

* Months_Inactive: As inactivity increases the attrition also increases, 2-4 months of inactivity are the biggest contributors of attrition -Bank can send automated messages to engage customers, these messages can be about their monthly activity, new offers or services, etc.

* Highest attrition is among the customers who interacted/reached out the most with/to the bank, This indicates that the bank is not able to resolve the problems faced by customers leading to attrition - a feedback collection system can be set up to check if the customers are satisfied with the resolution provided, if not, the bank should act upon it accordingly.