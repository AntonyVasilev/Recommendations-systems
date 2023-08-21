# Recommender system
**The aim:** creating a model to recommend the goods for the customers.  
**The metric:** 𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛@5

**Data (packed in data.zip):** 
 * retail_train.csv - a train dataframe with the data of the purchases
 * product.csv - an additional dataframe with an information about goods
 * hh_demographic.csv - a dataframe with an additional information about customers 
 * retail_test1.csv - a test dataframe with the data of the purchases
 
**The model:** Used a two-level model. CosineRecommender selects candidates for ranking at the first-level. CatBoostClassifier calculates scores for re-ranking candidates from 1st level at the second one.  

**The structure of the project:**  
 * **src.** functions: 
  * *metrics* - metrics, 
  * *utils* - data pre-filtering, 
  * *features* - creating new features gor the second level of the model, 
  * *recommenders* - contains a *MainRecommender* class which performs the selection of candidates and their re-ranking based on the score from the first-level model.  
 * **draft_notebooks.** draft notebooks with different solutions to the problem.  
 
### After re-ranking the model shows 𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛@5≈0.26324
