import csv
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import pandas as pd

import numpy as np


def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(
        np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


data = pd.read_csv('out.csv')
df = data
df.drop(columns=['Label'])
X = df
y = data['Label']
print(np.shape(X))
num_feats = 600
cor_support, cor_feature = cor_selector(X, y, num_feats)
# print(cor_support)
print(cor_feature)

print(str(len(cor_feature)), 'selected features')


# X_norm = MinMaxScaler().fit_transform(X)
# chi_selector = SelectKBest(chi2, k=num_feats)
# chi_selector.fit(X_norm, y)
# chi_support = chi_selector.get_support()
# # print(chi_support)
# chi_feature = X.loc[:, chi_support].columns.tolist()

# print(str(len(chi_feature)), 'selected features')


# # Recursive Feature Elimination

# rfe_selector = RFE(estimator=LogisticRegression(),
#                    n_features_to_select=num_feats, step=60, verbose=5)
# rfe_selector.fit(X_norm, y)
# rfe_support = rfe_selector.get_support()
# rfe_feature = X.loc[:, rfe_support].columns.tolist()
# print(str(len(rfe_feature)), 'selected features')

# #  Lasso: SelectFromModel

# embeded_lr_selector = SelectFromModel(
#     LogisticRegression(penalty="l2"), max_features=num_feats)
# embeded_lr_selector.fit(X_norm, y)

# embeded_lr_support = embeded_lr_selector.get_support()
# embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()
# print(str(len(embeded_lr_feature)), 'selected features')

# feature_name = X.columns.tolist()


# feature_selection_df = pd.DataFrame(
#     {'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support})
# # count the selected times for each feature
# feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# # display the top 100
# feature_selection_df = feature_selection_df.sort_values(
#     ['Total', 'Feature'], ascending=False)
# feature_selection_df.index = range(1, len(feature_selection_df)+1)
# # feature_selection_df.head(num_feats)

# # print(feature_selection_df)

# temp = list(feature_selection_df)
# print(temp)
# for i in feature_selection_df:
#     print(i)

# with open('innovators.csv', 'w', newline='') as file:
#     writer = csv.writer(file)

#     writer.writerow(temp)
