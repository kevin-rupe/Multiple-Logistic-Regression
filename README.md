# Multiple-Logistic-Regression
Project completed in pursuit of Master's of Science in Data Analytics.

## RESEARCH QUESTION 

For this assessment, I would like to determine which variables can accurately predict if the patient was readmitted after their initial stay at the hospital. For that reason, my research question is: 

***“What variables are highly correlated to the patient being readmitted to the hospital?”***

## GOALS

My goals for this performance assessment are simply stated: I want to gain more insight into which variables in our dataset might predict with accuracy why a patient is readmitted to the hospital.

## SUMMARY OF ASSUMPTIONS

Four assumptions of logistic regression models are as summarized first by stating that there must not be any collinearity between the independent variables chosen in your model. This is better stated that the independent variables cannot be too highly correlated with each other. Second, we assume that there is a linear relationship between the independent variables and log odds. Third, observations in our model should be selected from a large sample size. And lastly, the observations should be independent of each other (Statistics Solutions, 2024).

## TOOL BENEFITS

There are many benefits to using Python for data analysis. There are so many libraries that can be loaded that make analysis much easier than attempting to manually analyze the data. First, libraries such as Pandas, NumPy and Matplotlib allow for easy manipulation of data using statistics, data frames, and visualization (Pandas, 2023). Second, the statsmodels libraries perform many complicated regression techniques with just a few simple lines of code. So long as the analyst interprets the results correctly, you can rest assured that these libraries will give the most accurate results (Perktold, J., 2009-2023). 

## APPROPRIATE TECHNIQUE

Logistic Regression is an appropriate technique to analyze how certain variables in the dataset are correlated. Using Python libraries such as statsmodels can help you determine how closely the dependent variable is correlated to multiple explanatory variables. Using the Seaborn and Matplotlib libraries gives you a visual representation of how closely the variables are linearly related. Logistic Regression allows the analyst to predict with some certainty what level of effect the explanatory variables have on the dependent variable (Massaron, L., 2016). 

## DATA CLEANING GOALS

Before I can start logistic regression, I need to ensure the dataset is clean. To start, I will clean the data by finding any duplicate values using the duplicated( ) method which checks across rows for duplicates. I will treat any duplicate rows by dropping them from the dataset using the drop_duplicates Pandas command. I will also look for missing values using the isnull( ) and isna( ) functions. If any are found, I will treat these values by dropping them or using statistical imputation to impute the values. I will also treat outliers in the same way I treat missing values. Lastly, though not considered ‘data cleaning,’ I will wrangle categorical variables using re-expression techniques such as one-hot-encoding or Pandas’ dummy variables. 

Below is the annotated code that I used to clean the data.

```python
print(med_data.duplicated().value_counts())
print(med_data.isnull().value_counts())
print(med_data.isna().value_counts())
```
## DATA TRANSFORMATION

I chose several categorical variables and as such had to perform re-expression of these variables so they could be used in the predictive modeling. I created dummy variables on all the Categorical variables. The annotated code for these steps is below.

```python
prefix_list = ['ReAdmis','HighBlood', 'Overweight', 'Soft_drink', 'Stroke', 'Arthritis', 'Diabetes',
              'Hyperlipidemia', 'BackPain', 'Anxiety', 'Asthma', 'Allergic_rhinitis', 'Reflux_esophagitis',
              'Services', 'Initial_admin', 'Gender', 'Marital', 'Complication_risk']

med_data = pd.get_dummies(med_data, prefix=prefix_list, prefix_sep='_', dummy_na=False,                        
                          drop_first=True, columns=prefix_list)
```


## INITIAL MODEL

![Image 3-8-25 at 9 48 AM](https://github.com/user-attachments/assets/42a35f0f-68a7-4552-90ef-8e790b9dcc55)

## JUSTIFICATION OF MODEL REDUCTION

Before reducing the model, I checked the variables for multicollinearity using statsmodels variance_inflation_factor (VIF) package. I removed variables one-by-one, re-running VIF each time to re-check for multicollinearity. I removed Doc_visits, and VitD_levels, which both had values larger than 10.

```python
## Checking for Multicollinearity

# Import functions
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = med_data[['Age', 'VitD_levels', 'Doc_visits',
       'Full_meals_eaten', 'VitD_supp', 'Initial_days', 'ReAdmis_Yes', 'HighBlood_Yes',
       'Stroke_Yes', 'Arthritis_Yes',
       'Hyperlipidemia_Yes', 'BackPain_Yes', 'Anxiety_Yes',
       'Asthma_Yes', 'Allergic_rhinitis_Yes', 'Reflux_esophagitis_Yes',
       'Services_CT', 'Services_IV', 'Services_MRI', 'Initial_admin_Emergency',
       'Initial_admin_Observation', 'Gender_Male', 'Gender_Nonbinary',
       'Marital_Married', 'Marital_Single', 'Marital_Separated',
       'Marital_Widowed', 'Complication_risk_Low', 'Complication_risk_Medium']]

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)
```
![Image 3-8-25 at 9 49 AM](https://github.com/user-attachments/assets/5b3824ca-0bf4-4e43-bf90-b19443b75004)

To reduce the model, I used the backward stepwise elimination wrapper method of feature selection. Looking at the p-values in the regression summary table, I removed the largest p-value variable one at a time. For logistic regression, we can look at the LLR p-value and evaluate this metric for the entire model. Both the initial model and reduced model had a p-value of 0.000, however, I did remove variables based on their large p-values, but did end up leaving a couple of variables that I felt were relevant to the data I am analyzing. All in all, during this feature selection method I removed Gender_Nonbinary, all Marital variables, Initial_admin_Observation, Services_IV, VitD_supp, Full_meals_eaten, Age, and Complication_risk_Medium.

## REDUCED LOGISTIC REGRESSION MODEL

![Image 3-8-25 at 9 49 AM (1)](https://github.com/user-attachments/assets/976d98e1-d907-458f-81e5-40d506e52696)

## MODEL COMPARISON
  	
The Pseudo R-squared values in both the initial and reduced models were very high, 0.94. I chose to measure the AIC value in model comparison because I feel that despite several key metrics not shifting very much that the initial model is overfitted. The AIC gives insight into which model fits better. The reduced model AIC is 875.56 which is an improvement over the initial model whose AIC is 892.93. This shows that there is a much better level of fit in the reduced model’s independent variables jointly than with the kitchen sink approach in the initial model (Zach, 2021). 

## OUTPUT AND CALCULATIONS

```python
conf_matrix_all = model_ReAdmis_all.pred_table()
conf_matrix_red = model_ReAdmis_red.pred_table()
print(conf_matrix_all)
print(conf_matrix_red)
```
![Image 3-8-25 at 9 55 AM](https://github.com/user-attachments/assets/de9d4f7c-5a82-4bd4-b904-e1bfdd4539af)

```python
# Extract TN, TP, FN and FP from conf_matrix_all 
TN = conf_matrix_red[0,0]
TP = conf_matrix_red[1,1]
FN = conf_matrix_red[1,0]
FP = conf_matrix_red[0,1]

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + FN + FP + TP)
print("The accuracy    of the reduced model is", np.round((accuracy * 100),2), "%")

# Calculate and print the sensitivity
sensitivity = TP / (TP + FN)
print("The sensitivity of the reduced model is", np.round((sensitivity * 100),2), "%")

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("The specificity of the reduced model is", np.round((specificity * 100),2), "%")
```
> The accuracy    of the reduced model is 98.58 %
> 
> The sensitivity of the reduced model is 98.17 %
> 
> The specificity of the reduced model is 98.82 %

## REGRESSION EQUATION AND RESULTS OF ANALYSIS

The regression equation in my model is stated below:

> ln p1-p = -78.458 + 0.182 (Male) + 2.088 (Emergency Room) + 1.52 (CT Scan) + 2.64 (MRI) + 1.436 (Initial_days) + 0.805 (High BP) + 1.58 (Stroke) – 1.276 (Arthritis) – 0.299 (Allergies) – 0.381 (Reflux) – 1.315 (Asthma) – 1.069 (Anxiety) + 0.28 (Back Pain) + 0.29 (Hyperlipidemia) – 1.583 (Low Complication Risk)

The coefficients are interpreted as the log odds of the patient’s readmission. For this analysis, I am going to convert the coefficients from log odds to an odds ratio which in my opinion is an easier way to understand the results. Given all things constant, the odds of being readmitted to the hospital are 1.2 times more likely among males as compared to other genders. The odds ratio is calculated by taking ⅇcoⅇf (Boston University, 2013). 

A patient is 8.1 times more likely to be readmitted to the hospital if they were initially admitted via the Emergency Room compared to another means of admission; 4.57 times more likely if they had a CT Scan compared to other services received; 14 times more likely if they had an MRI compared to other services received; 4.2 times more likely per each day stayed at the hospital than staying one day less; 2.23 times more likely if the patient has high blood pressure; 4.85 times more likely is the patient had had a stroke; 3.58 times less likely if the patient has arthritis; 1.35 times less likely if the patient has allergic rhinitis; 1.46 times less likely if the patient has reflux esophagitis, 3.72 times less likely if the patient has asthma; 2.91 times less likely if the patient has anxiety; 1.32 times more likely if the patient has back pain; 1.34 times more likely if the patient has hyperlipidemia, and 4.87 times less likely to be readmitted to the hospital if the patient has a low complication risk compared to medium or high. 

This reduced model is statistically significant for several reasons. I removed insignificant p-values larger than 0.05 which means that all independent variables are statistically significant. I did choose to leave a few variables I thought were good for this model, and given the LLR p-value is 0.000 then I know my model shows a good fit. The reduced model performs well given that the Pseudo R-squared value is 0.9481. 

I believe this model is also practically significant for a hospital. This analysis would allow them to predict the number of patients that might be readmitted, and would allow the medical staff to particularly treat patients with more care that had these high-risk factors based on readmission. Hospitals get fined when the readmission rate is too high, so this data could provide useful financial confidence for investors and board members to show trends of how well they have improved year over year. 

The limitations of this analysis are in my performing backward stepwise elimination to reduce my model. The main limitation to this method is that some variables removed during this process may actually have a causal relationship to the dependent variable, and some variables that were left may have been coincidentally significant, but this method only looks at the p-values to determine significance. 

## RECOMMENDATIONS

I would recommend to the organization that they use the regression equation from my analysis to predict their company’s future trends in patient readmission. They could use this data to also focus more attention on certain high-risk factors to help drive down their overall readmission rate. Using this could help them be better prepared to avoid hefty fines incurred on their organization for having too many readmissions. This then could be provided to their shareholders and board members to promote a healthier financial confidence in the organization going forward. 

## SUPPORTING DOCUMENTATION

#### SOURCES FOR THIRD-PARTY CODE

Pandas (2023, June 28). Retrieved September 27, 2023, from https://pandas.pydata.org/docs/reference/index.html.

Perktold, J., Seabold, S., & Taylor, J. (2009-2023). Retrieved February 18, 2024, from https://www.statsmodels.org/stable/index.html. 

Waskom, M. (2012-2022). Seaborn Statistical Data Visualization. Retrieved September 27, 2023, from https://seaborn.pydata.org/index.html.

Zach. (2021, May 20). How to Calculate AIC of Regression Models in Python. Retrieved on February 18, 2024, from https://statology.org/aic-in-python/ 

#### SOURCES 

Statistic Solutions. (2024). Assumptions of Logistic Regression. Retrieved February 18, 2024, from https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-logistic-regression/.

Massaron, L. & Boschetti, A. (2016). Regression Analysis with Python: Learn the Art of Regression Analysis with Python. Packt Publishing.

Boston University School of Public Health. (2013, January 17). Multiple Logistic Regression Analysis. Retrieved February 18, 2024 from https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_multivariable/bs704_multivariable8.html 
