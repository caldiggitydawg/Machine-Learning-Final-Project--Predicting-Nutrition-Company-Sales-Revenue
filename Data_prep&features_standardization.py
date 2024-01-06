# Data cleaning and formatting file
# This is a prep file. The models are in 'matt_workthrough.py'

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
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
np.set_printoptions(suppress=True, threshold=5000, edgeitems=10)

# Print off basic stuff about our data set
print(my_df.sample(25))
print(my_df.info())

# This document will serve as the project journal. Every decision regarding the sourcing of the features,
# such as deleting nulls and dummy coding, will be accessible through cm_journal.pdf
# Appended as necessary and as the script develops
from reportlab.pdfgen import canvas

info_file = 'cm_journal.pdf'


def write_to_pdf(info_file, txt):
    pdf_canvas = canvas.Canvas(info_file)
    pdf_canvas.setFont("Helvetica", 10)
    y_coordinate = 800
    for line in txt:
        pdf_canvas.drawString(50, y_coordinate, line)
        y_coordinate -= 14
    pdf_canvas.save()

# deleting null values for dob and gender fields


desc = 'Cal Flett and Matt Timoney - Mattson Nutrition Reference Document'
my_df = my_df.dropna(subset=['dob', 'Gender'])
space = ' '
i_1 = f"(i) null values in 'dob' and 'gender' fields were dropped entirely; this decision was made because"
ii = f"(ii) the case background describes these nulls as 'significant omissions'. This reduces our number of total"
iii = f"(iii) observations to 66,161 ({66161-83216} or {((83216 - 66161)/83216)*100:2.2f}%)"

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

iv = f"(iv) 'Big 5' Categories were dummy-coded, resulting in 25 new fields broken down by response (1-5)"
v = f"(v) boolean expression is then converted to a binary, then the field is added to the features list"

# converted dob field format and created age variable by subtracting dob from extract date
# from datetime import datetime
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

vi = f"(vi) The BMI field was uniquely challenging because of the floats and nulls. We created weight bins to solve this"

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
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Sales', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']].values
target = my_df['delta1_Sales']
print(my_df.sample(10))

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(features_list)
features_sc = sc.transform(features_list)
pca = PCA(random_state=43543, n_components=5)
pca_fitted = pca.fit(features_sc)
pca_variance = pca_fitted.explained_variance_ratio_

vii = f"(vii) Our Principle Component Analysis with five components allows us to reduce our features significantly"
viii = f"(viii) PCA accuracies: '12.0%', '8.1%', '5.2%', '3.8%', '2.9%'"

features_names = ['Big5_Openness_1.0', 'Big5_Openness_2.0', 'Big5_Openness_3.0', 'Big5_Openness_4.0',
                'Big5_Openness_5.0', 'Big5_Extroversion_1.0', 'Big5_Extroversion_2.0', 'Big5_Extroversion_3.0',
                'Big5_Extroversion_4.0', 'Big5_Extroversion_5.0', 'Big5_Agreeableness_1.0', 'Big5_Agreeableness_2.0',
                'Big5_Agreeableness_3.0', 'Big5_Agreeableness_4.0', 'Big5_Agreeableness_5.0', 'Big5_Neuroticism_1.0',
                'Big5_Neuroticism_2.0', 'Big5_Neuroticism_3.0', 'Big5_Neuroticism_4.0', 'Big5_Neuroticism_5.0',
                'FamilyHistory_Diabetes_No', 'FamilyHistory_Diabetes_Yes', 'FamilyHistory_HeartDisease_No',
                'FamilyHistory_HeartDisease_Yes', 'FamilyHistory_Cancer_No', 'FamilyHistory_Cancer_Yes',
                'FamilyHistory_Crohns_No', 'FamilyHistory_Crohns_Yes', 'FamilyHistory_Alzheimer_No',
                'FamilyHistory_Alzheimer_Yes', 'FamilyHistory_Parkinsons_No', 'FamilyHistory_Parkinsons_Yes',
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Sales', 'delta2_Likes', 'delta2_Shares',
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
                'FamilyHistory_Other_No', 'FamilyHistory_Other_Yes', 'delta2_Sales', 'delta2_Likes', 'delta2_Shares',
                'delta2_WebsiteVisits', 'delta2_MobileAppLogins', 'delta2_Steps', 'delta2_DeepSleep',
                'delta2_LightSleep', 'delta2_REMSleep', 'delta2_HeartRate', 'delta2_Stress', 'CustomerAge',
                'Gender_Female','Gender_Male', 'Gender_Other', 'delta2_OfficeVisits_No', 'delta2_OfficeVisits_Yes',
                'delta2_BMI_large', 'delta2_BMI_medium', 'delta2_BMI_small', 'delta2_BMI_unknown',
                'delta2_Bloodpressure_High', 'delta2_Bloodpressure_Low', 'delta2_Bloodpressure_Normal',
                'delta2_Smoke_No', 'delta2_Smoke_Unknown', 'delta2_Smoke_Yes', 'delta2_Drink_Casual',
                'delta2_Drink_Excessive', 'delta2_Drink_None', 'delta2_Drink_Unknown']]

# --------------------------------------------------------------------------------------------------------------------
# plots and visualizations below

full_path = "FeaturesMN1.xlsx"
my_df.describe(include='all').to_excel(full_path)
corrMatrix = my_df.corr()
print(corrMatrix)
full_path = os.path.join(os.getcwd(), "feature_corr.xlsx")
corrMatrix.to_excel(full_path)


plt.close('all')
plt.figure(figsize=(50, 50))
sns.set(font_scale=2)
sns.heatmap(corrMatrix, annot=True)
plt.savefig(os.path.join(os.getcwd(), 'feature_corr.png'))
plt.close('all')


import seaborn as sns
import matplotlib.pyplot as plt

corr_vals = corrMatrix.values.flatten()
print(corr_vals)

plt.figure(figsize=(8, 6))
plt.hist(corr_vals, bins=75, color='skyblue', edgecolor='black')
plt.title('Feature Correlations')
plt.xlabel('Coefficients')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(os.getcwd(), 'corr_hist.png'))
plt.close('all')

# execute the following lines to produce a pdf document detailing our handling and formatting of the data
# prior to building models
txt = [desc, space, i_1, ii, iii, iv, v, vi, vii, viii]
write_to_pdf(info_file, txt)



