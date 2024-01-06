import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import os

# load in file
my_df = pd.read_json(os.path.join(os.getcwd(), 'Mattson_nutrition_customers.json'))

# housekeeping
import matplotlib
matplotlib.use('Agg')
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
np.set_printoptions(suppress=True, threshold=5000, edgeitems=10)

# Print off basic stuff about our data set
print(my_df.sample(25))
print(my_df.info())



# What to do with null values in our other fields?
# Better to dummy code them instead of deleting.
# Check for leakage, check null counts
print(my_df['CustomerID'].is_unique)
print(my_df.isnull().sum())

my_df = my_df.dropna(subset=['dob', 'Gender'])

big5_fields = ['Big5_Conscientiousness', 'Big5_Openness', 'Big5_Extroversion', 'Big5_Agreeableness',
               'Big5_Neuroticism']
features = []
for i in big5_fields:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))

familyhist_fields = ['FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease', 'FamilyHistory_Cancer',
                     'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer', 'FamilyHistory_Parkinsons',
                     'FamilyHistory_Other']

for i in familyhist_fields:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))

my_df = pd.concat([my_df, pd.get_dummies(my_df['Gender'], drop_first=False, prefix='Gender')], axis=1)
print(my_df.sample(25))
for i in features:

    my_df[i] = my_df[i].map({True: 1, False: 0})

# converted dob field format and created age variable by subtracting dob from extract date
from datetime import datetime
my_df['dob'] = pd.to_datetime(my_df['dob'], format='%Y%m%d').dt.strftime('%m/%d/%Y')
my_df['ExtractDate'] = pd.to_datetime(my_df['ExtractDate'])
my_df['dob'] = pd.to_datetime(my_df['dob'])
my_df['CustomerAge'] = (my_df['ExtractDate'] - my_df['dob']).dt.days / 365


my_df['delta1_Sales'] = my_df['delta1'].apply(lambda x: x.get('Sales'))

delta_1 = ['CustomerAge', 'Gender', 'delta1_Sales']

my_df['delta2_Sales'] = my_df['delta2'].apply(lambda x: x.get('Sales'))
my_df['delta2_OfficeVisits'] = my_df['delta2'].apply(lambda x: x.get('Medical', {}).get('OfficeVisits'))
my_df['delta2_BMI'] = my_df['delta2'].apply(lambda x: x.get('Medical', {}).get('BMI'))
my_df['delta2_Bloodpressure'] = my_df['delta2'].apply(lambda x: x.get('Medical', {}).get('Bloodpressure'))
my_df['delta2_Smoke'] = my_df['delta2'].apply(lambda x: x.get('LifeStyle', {}).get('Smoke'))
my_df['delta2_Drink'] = my_df['delta2'].apply(lambda x: x.get('LifeStyle', {}).get('Drink'))
my_df['delta2_Likes'] = my_df['delta2'].apply(lambda x: x.get('SocialMedia', {}).get('Likes'))
my_df['delta2_Shares'] = my_df['delta2'].apply(lambda x: x.get('SocialMedia', {}).get('Shares'))

delta_2 = ['delta2_Sales', 'delta2_OfficeVisits', 'delta2_BMI', 'delta2_Bloodpressure', 'delta2_Smoke',
           'delta2_Drink', 'delta2_Likes', 'delta2_Shares']
print(my_df.sample(25))
features.extend(list(delta_2))

my_df['delta2_WebsiteVisits'] = my_df['WebsiteVisits'].apply(lambda x: x.get('delta2'))
my_df['delta2_MobileAppLogins'] = my_df['MobileAppLogins'].apply(lambda x: x.get('delta2'))
my_df['delta2_Steps'] = my_df['Steps'].apply(lambda x: x.get('delta2'))
my_df['delta2_DeepSleep'] = my_df['Sleep'].apply(lambda x: x.get('delta2', {}).get('Deep'))
my_df['delta2_LightSleep'] = my_df['Sleep'].apply(lambda x: x.get('delta1', {}).get('Light'))
my_df['delta2_REMSleep'] = my_df['Sleep'].apply(lambda x: x.get('delta1', {}).get('REM'))
my_df['delta2_HeartRate'] = my_df['HeartRate'].apply(lambda x: x.get('delta2'))
my_df['delta2_Stress'] = my_df['Stress'].apply(lambda x: x.get('delta2'))

delta_list3 = ['delta2_WebsiteVisits', 'delta2_MobileAppLogins','delta2_Steps', 'delta2_DeepSleep',
               'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_stress']

features.extend(list(delta_list3))
features.extend(list(delta_1))

# it was at THIS POINT in the project where I realized that there would be leakage if we included any of the
# delta1 variables in the feature set, save the sales data.

# okay now that we have all of our features in a list, we've examined them by printing 'my_df', and found that BMI
# has some annoying nulls. in order to fill these and avoid deleting more data, we are going to perform the following:

my_df['delta2_BMI'] = my_df['delta2_BMI'].fillna(-1).astype(float)
print(my_df['delta2_BMI'].info())

def weight_class(a):
    if a < 0:
        return 'unknown'
    elif 19 >= a:
        return 'small'
    elif 19 <= a <= 25:
        return 'medium'
    else:
        return 'large'

bmi_sub_list = []
for i in my_df['delta2_BMI']:
    i = weight_class(i)
    bmi_sub_list.append(i)

my_df['delta2_BMI'] = bmi_sub_list

for c in my_df.select_dtypes(include=bool).columns:
    my_df[c] = my_df[c].map({True: 1, False: 0})

more_dummies = ['delta2_OfficeVisits', 'delta2_BMI', 'delta2_Bloodpressure', 'delta2_Smoke', 'delta2_Drink']

for i in more_dummies:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))
for c in my_df.select_dtypes(include=bool).columns:
    my_df[c] = my_df[c].map({True: 1, False: 0})
features = [x for x in features if x not in more_dummies]

for i in features:
    print(i)
print(my_df['Gender'])

features_list = my_df[['Big5_Openness_1.0', 'Big5_Openness_2.0', 'Big5_Openness_3.0', 'Big5_Openness_4.0',
                'Big5_Openness_5.0', 'Big5_Extroversion_1.0', 'Big5_Extroversion_2.0', 'Big5_Extroversion_3.0',
                'Big5_Extroversion_4.0', 'Big5_Extroversion_5.0', 'Big5_Agreeableness_1.0', 'Big5_Agreeableness_2.0',
                'Big5_Agreeableness_3.0', 'Big5_Agreeableness_4.0', 'Big5_Agreeableness_5.0', 'Big5_Neuroticism_1.0',
                'Big5_Neuroticism_2.0', 'Big5_Neuroticism_3.0', 'Big5_Neuroticism_4.0', 'Big5_Neuroticism_5.0',
                'FamilyHistory_Diabetes_No', 'FamilyHistory_Diabetes_Yes', 'FamilyHistory_HeartDisease_No',
                'FamilyHistory_HeartDisease_Yes', 'FamilyHistory_Cancer_No', 'FamilyHistory_Cancer_Yes',
                'FamilyHistory_Crohns_No', 'FamilyHistory_Crohns_Yes', 'FamilyHistory_Alzheimer_No',
                'FamilyHistory_Alzheimer_Yes', 'FamilyHistory_Parkinsons_No', 'FamilyHistory_Parkinsons_Yes',
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']].values
target = my_df['delta1_Sales']
print(target.shape)
print(features_list.shape)
print(my_df.sample(10))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(features_list)
features_sc = sc.transform(features_list)
pca = PCA(random_state=43543, n_components=5)
pca_fitted = pca.fit(features_sc)
pca_variance = pca_fitted.explained_variance_ratio_

features_names = ['Big5_Openness_1.0', 'Big5_Openness_2.0', 'Big5_Openness_3.0', 'Big5_Openness_4.0',
                'Big5_Openness_5.0', 'Big5_Extroversion_1.0', 'Big5_Extroversion_2.0', 'Big5_Extroversion_3.0',
                'Big5_Extroversion_4.0', 'Big5_Extroversion_5.0', 'Big5_Agreeableness_1.0', 'Big5_Agreeableness_2.0',
                'Big5_Agreeableness_3.0', 'Big5_Agreeableness_4.0', 'Big5_Agreeableness_5.0', 'Big5_Neuroticism_1.0',
                'Big5_Neuroticism_2.0', 'Big5_Neuroticism_3.0', 'Big5_Neuroticism_4.0', 'Big5_Neuroticism_5.0',
                'FamilyHistory_Diabetes_No', 'FamilyHistory_Diabetes_Yes', 'FamilyHistory_HeartDisease_No',
                'FamilyHistory_HeartDisease_Yes', 'FamilyHistory_Cancer_No', 'FamilyHistory_Cancer_Yes',
                'FamilyHistory_Crohns_No', 'FamilyHistory_Crohns_Yes', 'FamilyHistory_Alzheimer_No',
                'FamilyHistory_Alzheimer_Yes', 'FamilyHistory_Parkinsons_No', 'FamilyHistory_Parkinsons_Yes',
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']

for idx, row in enumerate(pca_fitted.components_):
    output = f'{100.0 * pca_variance[idx]:4.1f}%:    '
    for val, name in zip(row, features_names):
        if output.strip()[-1] == ":":
            output += f" {val:5.4f} * {name:s}"
        else:
            output += f" + {val:5.4f} * {name:s}"
    print(output)

my_df = my_df[['Big5_Openness_1.0', 'Big5_Openness_2.0', 'Big5_Openness_3.0', 'Big5_Openness_4.0',
                'Big5_Openness_5.0', 'Big5_Extroversion_1.0', 'Big5_Extroversion_2.0', 'Big5_Extroversion_3.0',
                'Big5_Extroversion_4.0', 'Big5_Extroversion_5.0', 'Big5_Agreeableness_1.0', 'Big5_Agreeableness_2.0',
                'Big5_Agreeableness_3.0', 'Big5_Agreeableness_4.0', 'Big5_Agreeableness_5.0', 'Big5_Neuroticism_1.0',
                'Big5_Neuroticism_2.0', 'Big5_Neuroticism_3.0', 'Big5_Neuroticism_4.0', 'Big5_Neuroticism_5.0',
                'FamilyHistory_Diabetes_No', 'FamilyHistory_Diabetes_Yes', 'FamilyHistory_HeartDisease_No',
                'FamilyHistory_HeartDisease_Yes', 'FamilyHistory_Cancer_No', 'FamilyHistory_Cancer_Yes',
                'FamilyHistory_Crohns_No', 'FamilyHistory_Crohns_Yes', 'FamilyHistory_Alzheimer_No',
                'FamilyHistory_Alzheimer_Yes', 'FamilyHistory_Parkinsons_No', 'FamilyHistory_Parkinsons_Yes',
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']]

# Machine Learning Models
# --------------------------------------------------------------------------------------------------------------------
# KNN, 43 neighbors best mix of accuracy plus variance
from sklearn.model_selection import train_test_split
f_train, f_test, t_train, t_test = train_test_split(features_list, target, test_size=0.20, random_state=55555)

from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler().fit(f_train)
f_train_sc = sc_train.transform(f_train)
sc_test = StandardScaler().fit(f_test)
f_test_sc = sc_test.transform(f_test)

from sklearn import neighbors
import math
num_neighbors = 43
knn = neighbors.KNeighborsRegressor(n_neighbors=num_neighbors, metric='euclidean', weights='uniform')
knn_model = knn.fit(f_train_sc, t_train)
score_test = 100 * knn_model.score(f_test_sc, t_test)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with test data = {score_test:.1f}%')
score_train = 100 * knn_model.score(f_train_sc, t_train)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with training data '
      f'to evaluate potential over-fitting = {score_train:.1f}%')

from sklearn.model_selection import GridSearchCV
hyper_parameters = {'n_neighbors': (15,20,25,30,35,40,45,50,55,60,65,70)}
knn = neighbors.KNeighborsRegressor()
clf = GridSearchCV(knn, hyper_parameters)
knn_model = clf.fit(f_train_sc, t_train)
scores = knn_model.cv_results_['mean_test_score']
print(scores)
score_test = 100 * knn_model.score(f_test_sc, t_test)
print(f'Grid search prediction accuracy with test data = {score_test:.1f}%')
score_train = 100 * knn_model.score(f_train_sc, t_train)
print(f'Grid search prediction accuracy with training data to evaluate potential '
      f'over-fitting = {score_train:.1f}%')




# Linear Regressor - Regular and SGD
from sklearn.linear_model import LinearRegression
lgr = LinearRegression(fit_intercept=True)
model = lgr.fit(f_train, t_train)
score_test = 100 * model.score(f_test, t_test)
print(f'Linear Regression Model Score with unseen testing data is {score_test:5.1f}%.')
score_train = 100 * model.score(f_train, t_train)
print(f'Linear Regression Model Score with training data to evaluate potential '
      f'over-fitting = {score_train:5.1f}%.')

from sklearn.ensemble import BaggingRegressor
bagged_regressor = LinearRegression(fit_intercept=True)
bagging_algorithm = BaggingRegressor(bagged_regressor, n_estimators=100)
bagged_ensemble_model = bagging_algorithm.fit(f_train, t_train)
score_test = 100 * bagged_ensemble_model.score(f_test, t_test)
score_train = 100 * bagged_ensemble_model.score(f_train, t_train)
print(f'Bagged regression model prediction accuracy with the unseen testing data is {score_test:.1f}%.')
print(f'Bagged regression model prediction accuracy with the unseen testing data is {score_train:.1f}%.')


from sklearn.linear_model import SGDRegressor
sgd_algorithm = SGDRegressor(fit_intercept=True, loss='squared_error',
                             learning_rate='adaptive', eta0=0.0001, penalty='l1',
                             max_iter=50000)
lgr_model = sgd_algorithm.fit(f_train_sc, t_train)
test_score = 100.0 * lgr_model.score(f_test_sc, t_test)
print(f'Linear regression model score with the unseen testing data is {test_score:.1f}%')
train_score = 100.0 * lgr_model.score(f_train_sc, t_train)
print(f'Linear regression model with the training data is {train_score:.1f}%')

from sklearn.model_selection import GridSearchCV
hyper_parameters = {"max_iter": [10000, 20000, 30000, 40000, 50000], "penalty": ['l1', 'l2'],
                    "learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive']}
grid_search = GridSearchCV(SGDRegressor(loss='squared_error', learning_rate='invscaling',
                                        eta0=0.001, max_iter=1000), hyper_parameters)
grid_model = grid_search.fit(f_train_sc, t_train)
print(grid_model.best_params_)
score_test = 100 * grid_model.score(f_test_sc, t_test)
print(f'Linear regression ({grid_model.best_params_}) prediction accuracy '
      f'with unseen testing data is {score_test:.1f}%.')




# Decision Tree, base parameters are good
from sklearn.tree import DecisionTreeRegressor
dtr_estimator = DecisionTreeRegressor(min_samples_leaf=5, min_samples_split=15, max_depth=10, ccp_alpha=0.1)
trained_dt_model = dtr_estimator.fit(f_train, t_train)
dt_model_score_test = trained_dt_model.score(f_test, t_test)
dt_model_score_train = trained_dt_model.score(f_train, t_train)
print(f'The base decision tree model had an accuracy score of {dt_model_score_test:,.2f} in'
      f' the unseen testing data.')
print(f'This model had a score of {dt_model_score_train:,.2f} in the training data, '
      f'so the variance was {(dt_model_score_train-dt_model_score_test):,.2f}')

from sklearn.model_selection import GridSearchCV
hyper_parameters = {"max_depth": [1, 5, 10], "min_samples_split": [10, 20, 30],
                    "min_samples_leaf": [5, 10, 15],
                    "ccp_alpha": [0, 0.1, 0.01]}
grid_search = GridSearchCV(DecisionTreeRegressor(), hyper_parameters)
gs_dt_model = grid_search.fit(f_train, t_train)
gs_dt_model_score_test = gs_dt_model.score(f_test, t_test)
gs_dt_model_score_train = gs_dt_model.score(f_train, t_train)
print(f'The decision tree model with gridsearch had an accuracy score of {gs_dt_model_score_test:,.2f} in'
      f' the unseen testing data. This model had a score of {gs_dt_model_score_train:,.2f} in the testing data, '
      f'so the variance was {(gs_dt_model_score_train-gs_dt_model_score_test):,.2f}')



# Random Forest
from sklearn.ensemble import RandomForestRegressor
hyper_params = {'bootstrap': True, 'max_samples': 1000, 'oob_score': True, 'max_features': 'sqrt',
                'criterion': 'squared_error', 'n_estimators': 100, 'random_state': 55555,
                'min_samples_leaf': 5, 'min_samples_split': 30, 'max_depth': 10}
my_classifier = RandomForestRegressor(**hyper_params)
rf_model = my_classifier.fit(f_train, t_train)
rf_model_score_test = rf_model.score(f_test, t_test)
rf_model_score_train = rf_model.score(f_train, t_train)
print(f'The base random forest model had an accuracy score of {rf_model_score_test:,.2f} in'
      f' the unseen testing data.')
print(f'The model had a score of {rf_model_score_train:,.2f} in the training data, '
      f'so the variance was {(rf_model_score_test-rf_model_score_train):,.2f}')



# Support Vector Machine - Takes too long to run normally, switched to PCA
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
features_reduced = pca_fitted.fit_transform(features_sc)
f_train, f_test, t_train, t_test = train_test_split(features_reduced, target, test_size=0.20, random_state=55555)
k = 'linear'
svm_algorithm = svm.SVR(kernel='linear', C=1, epsilon=0.2)
svm_model = svm_algorithm.fit(f_train, t_train)
predicted = svm_model.predict(f_test)
mse_test = mean_squared_error(t_test, predicted)
r2_test = r2_score(t_test, predicted)
print(f'SVM using {k} kernel had Mean Squared Error on testing data: {mse_test:.2f}')
print(f'SVM using {k} kernel had R-squared on testing data: {r2_test:.2f}')
predicted_train = svm_model.predict(f_train)
mse_train = mean_squared_error(t_train, predicted_train)
r2_train = r2_score(t_train, predicted_train)
print(f'SVM using {k} kernel had Mean Squared Error on training data: {mse_train:.2f}')
print(f'SVM using {k} kernel had R-squared on training data: {r2_train:.2f}')


# Model Implementation
# --------------------------------------------------------------------------------------------------------------------
# Saving Model to pkl
import joblib
filename = (os.path.join(os.getcwd(), 'Decision_Tree.pkl'))
with open(filename, 'wb') as fout:
    joblib.dump(trained_dt_model, fout)

with open(filename, 'rb') as fin:
    dt_pkl = joblib.load(fin)

# Preparing Sample Text
filename = (os.path.join(os.getcwd(), 'sample_implementation.txt'))
my_df = pd.read_csv(filename, delimiter='|')
print(my_df.info())
my_df = my_df.dropna(subset=['dob', 'Gender'])
big5_fields = ['Big5_Conscientiousness', 'Big5_Openness', 'Big5_Extroversion', 'Big5_Agreeableness',
               'Big5_Neuroticism']
features = []
for i in big5_fields:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))
familyhist_fields = ['FamilyHistory_Diabetes', 'FamilyHistory_HeartDisease', 'FamilyHistory_Cancer',
                     'FamilyHistory_Crohns', 'FamilyHistory_Alzheimer', 'FamilyHistory_Parkinsons',
                     'FamilyHistory_Other']
for i in familyhist_fields:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))
my_df = pd.concat([my_df, pd.get_dummies(my_df['Gender'], drop_first=False, prefix='Gender')], axis=1)
print(my_df.sample(25))
for i in features:
    my_df[i] = my_df[i].map({True: 1, False: 0})
print(my_df.info())
print(my_df.sample(1))
my_df['delta2_Likes'] = my_df['SocialMediaLikes']
my_df['delta2_Shares'] = my_df['SocialMediaShares']
my_df['delta2_WebsiteVisits'] = my_df['WebsiteVisits']
my_df['delta2_Likes'] = my_df['SocialMediaLikes']
my_df['delta2_MobileAppLogins'] = my_df['MobileAppLogins']
my_df['delta2_Steps'] = my_df['Steps']
my_df['delta2_DeepSleep'] = my_df['SleepDeep']
my_df['delta2_LightSleep'] = my_df['SleepLight']
my_df['delta2_REMSleep'] = my_df['SleepREM']
my_df['delta2_Stress'] = my_df['Stress']
my_df['delta2_HeartRate'] = my_df['HeartRate']
my_df['delta2_Likes'] = my_df['SocialMediaLikes']


my_df['delta2_BMI'] = my_df['BMI'].fillna(-1).astype(float)
print(my_df['delta2_BMI'].info())
def weight_class(a):
    if a < 0:
        return 'unknown'
    elif 19 >= a:
        return 'small'
    elif 19 <= a <= 25:
        return 'medium'
    else:
        return 'large'
bmi_sub_list = []
for i in my_df['delta2_BMI']:
    i = weight_class(i)
    bmi_sub_list.append(i)
my_df['delta2_BMI'] = bmi_sub_list
for c in my_df.select_dtypes(include=bool).columns:
    my_df[c] = my_df[c].map({True: 1, False: 0})
more_dummies = ['OfficeVisits', 'delta2_BMI', 'BloodPressure', 'Smoke', 'Drink']
for i in more_dummies:
    dummies = pd.get_dummies(my_df[i], drop_first=False, prefix=i, dummy_na=True)
    my_df = pd.concat([my_df, dummies], axis=1)
    features.extend(list(dummies.columns))
for c in my_df.select_dtypes(include=bool).columns:
    my_df[c] = my_df[c].map({True: 1, False: 0})
features = [x for x in features if x not in more_dummies]

my_df['delta2_OfficeVisits_No'] = my_df['OfficeVisits_No']
my_df['delta2_OfficeVisits_Yes'] = my_df['OfficeVisits_Yes']
my_df['delta2_Bloodpressure_High'] = my_df['BloodPressure_High']
my_df['delta2_Bloodpressure_Low'] = my_df['BloodPressure_Low']
my_df['delta2_Bloodpressure_Normal'] = my_df['BloodPressure_Normal']
my_df['delta2_Smoke_No'] = my_df['Smoke_No']
my_df['delta2_Smoke_Unknown'] = my_df['Smoke_Unknown']
my_df['delta2_Smoke_Yes'] = my_df['Smoke_Yes']
my_df['delta2_Drink_Casual'] = my_df['Drink_Casual']
my_df['delta2_Drink_Excessive'] = my_df['Drink_Excessive']
my_df['delta2_Drink_None'] = my_df['Drink_nan']
my_df['delta2_Drink_Unknown'] = my_df['Drink_Unknown']
from datetime import datetime
def calculate_age(dob):
    birth_date_obj = datetime.strptime(str(dob),  '%Y%m%d.%f',)
    current_date = datetime.now()
    age = current_date.year - birth_date_obj.year - ((current_date.month, current_date.day) < (birth_date_obj.month, birth_date_obj.day))
    return age
my_df['CustomerAge'] = my_df['dob'].apply(lambda x: calculate_age(x))

features_list = my_df[['Big5_Openness_1.0', 'Big5_Openness_2.0', 'Big5_Openness_3.0', 'Big5_Openness_4.0',
                'Big5_Openness_5.0', 'Big5_Extroversion_1.0', 'Big5_Extroversion_2.0', 'Big5_Extroversion_3.0',
                'Big5_Extroversion_4.0', 'Big5_Extroversion_5.0', 'Big5_Agreeableness_1.0', 'Big5_Agreeableness_2.0',
                'Big5_Agreeableness_3.0', 'Big5_Agreeableness_4.0', 'Big5_Agreeableness_5.0', 'Big5_Neuroticism_1.0',
                'Big5_Neuroticism_2.0', 'Big5_Neuroticism_3.0', 'Big5_Neuroticism_4.0', 'Big5_Neuroticism_5.0',
                'FamilyHistory_Diabetes_No', 'FamilyHistory_Diabetes_Yes', 'FamilyHistory_HeartDisease_No',
                'FamilyHistory_HeartDisease_Yes', 'FamilyHistory_Cancer_No', 'FamilyHistory_Cancer_Yes',
                'FamilyHistory_Crohns_No', 'FamilyHistory_Crohns_Yes', 'FamilyHistory_Alzheimer_No',
                'FamilyHistory_Alzheimer_Yes', 'FamilyHistory_Parkinsons_No', 'FamilyHistory_Parkinsons_Yes',
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']].values
model_predictions = dt_pkl.predict(features_list)
my_df['predicted_value'] = pd.Series(model_predictions)
file = 'predictions.xlsx'
my_df = my_df.dropna(subset=['predicted_value'])
my_df[['CustomerAge', 'Gender_Female','Gender_Male', 'Gender_Other','delta2_Likes', 'delta2_Shares','delta2_WebsiteVisits',
        'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep','delta2_LightSleep', 'delta2_REMSleep',
        'delta2_Stress', 'predicted_value']].to_excel(file, index=False)