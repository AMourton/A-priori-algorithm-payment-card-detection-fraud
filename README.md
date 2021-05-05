# A-priori-algorithm-payment-card-detection-fraud
Implementation of a priori algorithm for credit card fraud detection

Files included:
 -- apriori.py
 -- inputs.csv --> the raw dataset in csv format.
 -- inputs1.xls --> dataset sample after initial pre-processing.
 -- tab2.csv --> the pre-processed dataset in csv format.

In this work- part of a personal University course project- a real- world dataset is used, namely UCSD- FICO Data mining contest 2009. The labelled dataset consists of 100,000 e- commerce transactions, organised in 20 anonymous fields. 
The algorithm is built on Visual Studio IDE for Python version 3.7.7 for 64- bit Mac OS.

1.	The raw data
In this work, a real- world dataset is used, namely UCSD- FICO Data mining contest 2009. The labelled dataset consists of 100,000 e- commerce transactions, organised in 20 anonymous fields. 


2.	Data pre-processing
After careful examination of the data, it was concluded that some of the fields have identical values, thus the repeated ones could be removed in order to reduce the dimensionality of the dataset. The attributes ‘amount’ and ‘total’, ‘hour1’ and ‘hour2’ have the same values, therefore only one could be kept. The attributes ‘custAttr1’ and ‘custAttr2’ are unique fields for each customer, the first being the card number of the customer and the second- the email address. ‘cutstAttr1’ was removed and ‘custAttr2’ was kept in the dataset to be used as the identification field for each customer. A new field called ‘transid’ with values 1- 99,999 was added to each row in the dataset to be used as id for each card transaction. Therefore, the final dataset consists of 18 fields.


3.	Inspection of the data
After finishing the data pre- processing, the data was carefully reviewed and analysed so that a better understanding on the potential of the information available in the database. The following statistical information was stored in the dataset:
Number of payment card transactions by state (including insular areas or US military mail code)- the top five USA states by number of transactions are California (19,717), Florida (8055), Texas (6785), New York (6566) and Georgia (4105). At the bottom of the ranking are Vermont (106), North Dakota (105), U.S. Armed Forces- Pacific (19), U.S. Armed Forces- Europe (3), Puerto Rico (1). See Appendix 2. 
Total amount of card transactions by state (including insular areas or US military mail code)- the top five USA states by total amount of card transactions are California (573,733.42), Florida (220,006.27), Texas (187,057.13), New York (182,676.18) and Georgia (109,211.99). At the bottom of the ranking are Wyoming (2,835.61), North Dakota (2,654.93), U.S. Armed Forces- Pacific (562.51), U.S. Armed Forces- Europe (101.75), Puerto Rico (49.95). 


4.	Create training and testing datasets
The quality of data is key to a good machine learning model. This is why it is very important that the training data to be clean and balanced. Imbalanced data refers to a classification problem where the classes are not represented equally. The pre- processed UCSD- FICO Data mining contest 2009 dataset is highly imbalanced, with the fraud transactions class representing only 2.65% of all transactions. This can lead to overfitting of the model which would result in unrealistically high accuracy rate. One of the approaches to tackle this problem is resampling. Resampling is performed after the dataset is split into training and testing data. Resampling is done only on the training dataset. There are two types of resampling: under sampling and oversampling. In under sampling, records are removed from the majority class and in over sampling records from the minority class are duplicated. In this work, the following steps are performed on the dataset to create the model:
Clients with only one transaction- these are removed from the initially pre-processed dataset as they do not have representing statistical value for the model. 44, 250 out of 100,000 records are left in the dataset, where the number of unique clients is 14,374.

5.	Create legal and fraudulent pattern
From both training and testing datasets, a legal and fraudulent pattern are created for each customer in the card payments database. This would allow the fraud detector to match each incoming transaction to one of the two patterns. The patterns are created using frequent itemset mining. An itemset is frequent if it appears simultaneously in as many transactions as the user defined minimum support. We assume that there is a number s, called support threshold and is defined as the fraction of records of database D that contains the itemset I as a subset. The support is calculated by the formula:

support(I) = count (I)
	           |D|

For example, if the dataset D contains 5,000 transactions where the itemset I appears 50 times, the support for this itemset will be 0.01 or 1%.
Researchers have found that fraudsters behave the same or similar way to genuine cardholders, thus it is better to create both fraud and legal pattern for each client, rather than finding a common pattern for fraudsters. This way, when frequent itemset mining is applied to a credit card transaction belong to a specific client, it would return the same set of attributes with the same values as those in the group of transactions specified by the support. Frequent mining techniques like A priori algorithm returns many such groups and the group consisting of the highest number of attributes determines the client’s behaviour pattern. The pattern recognition is built as follows:
1) The dataset is grouped by individual client with his associated transactions. Each transaction is given an individual id number too.

2) Each client’s transactions are grouped by their class label- fraudulent or legitimate.

3) Apply a priori algorithm to the legitimate dataset to find the frequency of the items dataset for each customer. The minimum support is set to 0.9. The largest itemset contains the following attributes: zip, amount, hour, flag5, field4. There are 20153 out of 21096 transactions matching the legit pattern and it is saved as such in the database. 

4) Apply a priori algorithm to the fraudulent dataset to find the frequency of the items dataset for each customer. The minimum support is set to 0.9. The largest itemset contains the following attributes: zip, hour, field1, flag5, Class and field4. There are 950 out of 1029 transactions matching the fraudulent pattern and it is saved as such in the database.

  


