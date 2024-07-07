# Machine Learning Using Decision Tree: AllLife Bank Personal Loan Campaign


## Problem Statement

### Context

AllLife Bank is a US bank that has a growing customer base. The majority of these customers are liability customers (depositors) with varying sizes of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors).

A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio.

You as a Data scientist at AllLife bank have to build a model that will help the marketing department to identify the potential customers who have a higher probability of purchasing the loan.


### Objective

To predict whether a liability customer will buy personal loans, to understand which customer attributes are most significant in driving purchases, and identify which segment of customers to target more.


### Data Dictionary
* `ID`: Customer ID
* `Age`: Customer’s age in completed years
* `Experience`: #years of professional experience
* `Income`: Annual income of the customer (in thousand dollars)
* `ZIP Code`: Home Address ZIP code.
* `Family`: the Family size of the customer
* `CCAvg`: Average spending on credit cards per month (in thousand dollars)
* `Education`: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
* `Mortgage`: Value of house mortgage if any. (in thousand dollars)
* `Personal_Loan`: Did this customer accept the personal loan offered in the last campaign? (0: No, 1: Yes)
* `Securities_Account`: Does the customer have securities account with the bank? (0: No, 1: Yes)
* `CD_Account`: Does the customer have a certificate of deposit (CD) account with the bank? (0: No, 1: Yes)
* `Online`: Do customers use internet banking facilities? (0: No, 1: Yes)
* `CreditCard`: Does the customer use a credit card issued by any other Bank (excluding All life Bank)? (0: No, 1: Yes)



## Exploratory Data Analysis.

- EDA is an important part of any project involving data.
- It is important to investigate and understand the data better before building a model with it.
- A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
- A thorough analysis of the data, in addition to the questions mentioned below, will help drive the EDA.



**Answers to relevant questions:**:

1. What is the distribution of mortgage attribute? Are there any noticeable patterns or outliers in the distribution?
    * There are quite a few outliers in the data.
    * However, we will not treat them as they are proper values.
    * 75% of the mortgage data have 0 values indicatings the customers does not have house mortgage payment.
    * Mortgage have a heavily right skewed distributions.

2. How many customers have credit cards?
    * 1470 customers have credit cards

3. What are the attributes that have a strong correlation with the target attribute (personal loan)?
    * Income attribute has the strongest correlation (positive) with the target attribute (personal loan)

4. How does a customer's interest in purchasing a loan vary with their age?
    * It does not, customers age plays little or no significance in purchasing a loan.
    * The average age of customers who purchased a loan and those that did not purchase a loan are similar.

5. How does a customer's interest in purchasing a loan vary with their education?
    * Customers with Advanced/Professional Education are the most likely group to purchase a loan.
    * Around 81% of customers who purchased a loan are Graduates or have Advanced/Professional education.






### Model Evaluation Criterion

#### Model can make wrong predictions as:

* Predicting a liability customer will not buy personal loans but in reality the customer will buy (FN).
* Predicting a liability customer will buy a personal loans but in reality the customer will not buy (FP).

#### Which case is more important?
* If we predict that a liability customer will not buy personal loan, but in reality the customer buys, the bank would lose an opportunity of providing loans to a potential customer.
* If we predict that a liability customer will buy, and they end up not buying, the bank would bear the cost of marketing campaign.
* The cost of marketing campaign is generally less compared to the interest of loans lost to a potential customer.

#### How to reduce loses?
* `recall` should be maximized, the greater the recall, the higher the chances of minimizing false negatives.









|--- Income <= 98.50
|   |--- CCAvg <= 2.95
|   |   |--- weights: [1362.83, 0.00] class: 0
|   |--- CCAvg >  2.95
|   |   |--- weights: [70.80, 93.75] class: 1
|--- Income >  98.50
|   |--- Family <= 2.50
|   |   |--- weights: [285.95, 739.58] class: 1
|   |--- Family >  2.50
|   |   |--- weights: [30.42, 916.67] class: 1



# Training performance comparison

            DecTree without class_weight	        DecTree with class_weight	        Decision Tree (Pre-Pruning)	        Decision Tree (Post-Pruning)
Accuracy	1.00000	                                    1.00000		                        0.80000			                        0.93600
Recall	    1.00000		                                1.00000		                        1.00000			                        1.00000
Precision	1.00000	                                	1.00000		                        0.32432		                        	0.60000
F1	        1.00000		                                1.00000		                        0.48980		                        	0.75000




# Testing performance comparison
            DecTree without class_weight	        DecTree with class_weight	        Decision Tree (Pre-Pruning)	        Decision Tree (Post-Pruning)
Accuracy	0.98133	                                    0.98067	                            0.81600	                                0.93133
Recall	    0.86111	                                    0.87500		                        1.00000		                            0.98611
Precision	0.93939	                                    0.91971	                        	0.34286		                            0.58436
F1	        0.89855	                                    0.89680		                        0.51064		                            0.73385



* The Decision Tree (Pre-Pruning) models has the highest `recall` scores on both training and test sets.
* We will choose the Decision Tree (Pre-Pruning) as the best model since our objective is to maximize `recall`. The greater the recall, the higher the chances of minimizing false negatives.