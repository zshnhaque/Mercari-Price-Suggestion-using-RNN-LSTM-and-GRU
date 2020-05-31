# Mercari Price Prediction using RNN, GRU and LSTM.

## Introduction

Mercari is a marketplace where one can sell or buy almost anything. The user (seller) list item in minutes and buyer can easily buy it. In this Project, we will predict the sale price of a listing based on information a user provides for this listing. The dataset is taken from Kaggle and Mercari Price Prediction was a challenge.

## Variable Description 
1.	train_id or test_id - the id of the listing 
2.	name - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. 
3.	item_condition_id - the condition of the items provided by the seller 
4.	category_name - category of the listing 
5.	brand_name – Brand name of the product
6.	price - the price that the item was sold for. This is the target variable that we will predict. The unit is USD. (Target variable)
7.	shipping - 1 if shipping fee is paid by seller and 0 by buyer 
8.	item_description - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage.
## Problem Definition and What to Predict
For a product listing based on information a user provides, we have to predict the sale price of the product using predictive model. Since we have to predict a continuous variable, this project is form of ‘Regression’

## Summary of Dataset
Training dataset sample – 1,482,535 and Features – 8
Unseen (Test) Dataset sample – 693359 and Features - 7

## Data Preprocessing
# Data Cleaning of missing values or NAs
 
Three features ‘category_name’, ‘brand_name’ and ‘item_description’ consists of Null-value or NA and we treat missing values as ‘missing’ , ‘no brand name’ and ‘no item description’ respectively. 

# Encoding
The categorical variables are transformed to factors using label encoding, since we have high cardinality for features like ‘category_name’, and one-hot encoding will cause high-dimensionality.

# Text features to Word Embedding for features ‘name’ and ‘item_description.’

Reason to choose ‘Word Embedding method’ instead of traditional ways like TF-IDF, because Word Embedding converts a word to an n-dimensional vector. Words which are related such as ‘Shoes’ and ’Socks’ map to similar n-dimensional vectors, while dissimilar words such as ‘Shoes’ and ‘Jewellery’ have dissimilar vectors. In this way the ‘meaning’ of a word can be reflected in its embedding, a model is then able to use this information to learn the relationship between words. The benefit of this method is that a model trained on the word ‘house’ will be able to react to the word ‘home’ even if it had never seen that word in training.
 

1)	Cleaning the text rows by removing stop words. Performing stemming and lemmatization using NLTK library.
2)	Converting the cleaned text into tokenized vectors using keras.preprocessing library.

3)	Next step is to get maximum length of the features and embed each row in a constant length which is required in CNN architecture.

# Log transformation of Target Variable

 



# Sequential Model Architecture
The idea behind RNNs is to make use of sequential information. In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for our language modelling tasks, traditional methods are erroneous. If we want to predict the next word in a sentence we better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Given a sequence of words we want to predict the probability of each word given the previous words. RNN allow us to measure how likely a sentence is, which is an important input for Machine Translation (since high-probability sentences are typically correct). 

The sequential model is built using Keras with Tensorflow Backend.
Summary of Architecture – 
1)	At first, we define the input features which are to be used for the deep learning model.
2)	Following to this, we initialize the embedding transformation of features containing text only (name, brand_name,category_name (converted to general_category, subcategory1, subcategory2) and item_description)
3)	Next, we create keras sequential model where we initialize ‘RNN’ layer for text feature (name and item_description). Along with this we add dropout regularization. 
4)	In case of GRU and LSTM, we update the keras.layer function.
5)	Lastly, we add model optimizer comprising of Loss fuction and Evaluation metric. Since it is regression, Evaluation metric is ‘MSE’ (mean squared error) and we use ‘MAE’ (mean absolute error) separately. Loss function is ‘adam’. 
6)	We split the main dataset into 70:15:15 for training, validation and testing.
7)	Training size – 1,037,774 , validation size – 222,380 , test size – 222,380

# Model Run and Evaluation

We continuously check the training loss (mse) and validation loss (mse), with more epoch(s) when training loss keeps on decreasing and Validation loss increases, we stop our model running process.
We use evaluation metric – MAE (Mean Absolute Error)



# Observation and Further Improvements
Initially, the model with traditional RNN layer, the Mean Absolute Error on Development Test Data is 55.8% but after changing the architecture to GRU and LSTM. It was observed that using GRU, the Mean Absolute Error on Development Test Data is 48.7 % and using LSTM, the Mean Absolute Error on Development Test Data is 48.55 %, so LSTM performs better out of all model architecture. Therefore, with this project we can say LSTM gives us the most Control-ability and thus, better results, but also comes with more Complexity and computation cost.

# Further Work or Improvement.
•	We can achieve better results if we implement BatchNormalisation, this can be done before the linear transformation happening at output layer.
•	We can check performance by increasing embedding factors.


# Conclusion
This is a project based on Artificial Neural Network because the project involves a lot of feature engineering and data preprocessing along with various deep learning architecture comprising of GRU and LSTM. The prediction on the Product list may not be accurate but the project has proved that predicting on Price ‘Target’ having only categorical and text features can be achieved using word embedding and cleaning the features containing texts using natural language processing techniques and this is the good starting point to understand how deep learning architecture plays a major role in getting a stable model. Comparing the evaluation results in this project we can say LSTM gives us the most Control-ability and thus, better results, but also comes with more Complexity and computation cost.


