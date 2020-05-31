Mercari Price Prediction.
In this Project, we will predict the sale price of a listing based on information a user provides for this listing.
Variable Description 
1.	train_id or test_id - the id of the listing 
2.	name - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. 
3.	item_condition_id - the condition of the items provided by the seller 
4.	category_name - category of the listing 
5.	brand_name •	
6.	price - the price that the item was sold for. This is the target variable that we will predict. The unit is USD. (Target variable)
7.	shipping - 1 if shipping fee is paid by seller and 0 by buyer 
8.	item_description - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage.
Problem Definition and What to Predict
For a product listing based on information a user provides, we have to predict the sale price of the product using predictive model. Since we have to predict a continuous variable, this project is form of ‘Regression’

Summary of Dataset
Training dataset sample – 1,482,535 and Features – 8
Unseen (Test) Dataset sample – 693359 and Features - 7


Data Preprocessing

Data Cleaning of missing 

 
Three features ‘category_name’, ‘brand_name’ and ‘item_description’ consists of Null-value or NA and we treat missing values as ‘missing’ , ‘no brand name’ and ‘no item description’ respectively. 


The categorical variables are transformed to factors using label encoding, since we have high cardinality for features like ‘category_name’, and one-hot encoding will cause high-dimensionality.
For Text varibles/features like 
Target to log transform

Text features to Word Embedding for features ‘name’ and ‘item_description.’
Reason to choose ‘Word Embedding method’ instead of traditional ways like TF-IDF, bcause Word Embedding converts a word to an n-dimensional vector. Words which are related such as ‘Shoes’ and ’Socks’ map to similar n-dimensional vectors, while dissimilar words such as ‘Shoes’ and ‘Jeweleery’ have dissimilar vectors. In this way the ‘meaning’ of a word can be reflected in its embedding, a model is then able to use this information to learn the relationship between words. The benefit of this method is that a model trained on the word ‘house’ will be able to react to the word ‘home’ even if it had never seen that word in training.
 

1)	Cleaning the text rows by removing stop words. Performing stemming and lemmatization using NLTK library.
2)	Converting the cleaned text into tokenized vectors using keras.preprocessing library.
 
It is observed that cleaned ‘item_description’ is transformed to tokenized sequential ‘seq_item_description’.
3)	Next step is to get maximum length of the features and embed each row in a constant length which is required in CNN architecture.

Sequential Model Architecture
The sequential model is built using Keras with Tensorflow Backend.
Summary of Architecture – 
1)	At first, we define the input features which are to be used for the deep learning model.
2)	Following to this, we initialize the embedding transformation of features containing text only (name, brand_name,category_name (converted to general_category, subcategory1, subcategory2) and item_description)
3)	Next, we create keras sequential model where we initialize ‘RNN’ layer for text feature (name and item_description). Along with this we add dropout regularization. 
4)	Lastly, we add model optimizer comprising of Loss fuction and Evaluation metric. Since it is regression, Evaluation metric is ‘MSE’ (mean squared error) and we use ‘MAE’ (mean absolute error) separately. Loss function is ‘adam’. 

Model Evaluation

Observation and Further Improvements
Initially, the model with traditional RNN layer, the Mean Absolute Error on Development Test Data is 55.8% but after changing the architecture to GRU and LSTM. It was observed that using GRU, the Mean Absolute Error on Development Test Data is 48.7 % and using LSTM, the Mean Absolute Error on Development Test Data is 48.55 %, so LSTM performs better out of all model architecture. Therefore, with this project we can say LSTM gives us the most Control-ability and thus, better results, but also comes with more Complexity and computation cost.
Further Work or Improvement.
BatchNormalisation


Conclusion
This is a very exciting project based on Artificial Neural Network because the project involves a lot of feature engineering and data preprocessing along with various deep learning architecture comprising of GRU and LSTM. The prediction on the Product list may not be accurate but the project has proved that predicting on Price ‘Target’ having only categorical and text features can be achieved using word embedding and cleaning the features containing texts using natural language processing techniques and this is the good starting point to understand how deep learning architecture plays a major role in getting a stable model. . Comparing the evaluation results in this project we can say LSTM gives us the most Control-ability and thus, better results, but also comes with more Complexity and computation cost.


Github Link - 

