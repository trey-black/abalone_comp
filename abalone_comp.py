# Create a set of histograms to investigate skew
# Come up with a baseline success rate
# Note sex having a third category
# Set up the processing pipeline
# Run a correlation matrix

# Should I convert y to a categorical variable?
# I guess I could run this as both a regression problem and a categorical problem, so might be worth setting up infrastructure for both?
# Let's start with categorical and then run everything for numerical

import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

train = pd.read_csv('/Users/treyb/Documents/Data Science/Programs/abalone_comp/train.csv', index_col=0)
train_x = train.drop(['Rings'], axis=1)
train_y = train['Rings'].astype('object')

# ### EDA ###
train.hist()
plt.show()

# Calculate skew (and compare to a normal dist)
# Calculate kurtosis
# Shapiro-Wilk test
# Q-Q plots

# # train_df.info()
# # No nulls in the data, sex needs to be encoded, numerical data needs to be standardized

# ### DATA PROCESSING PIPELINE ###
# num_features = train.select_dtypes(include=['int64', 'float64']).columns
# cat_features = train.select_dtypes(include=['object']).columns

# num_trans = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# cat_trans = Pipeline(steps=[
#     ('encoder', OrdinalEncoder())
# ])

# preprocessor = ColumnTransformer(
#     transformers = [
#         ('numeric', num_trans, num_features),
#         ('categorical', cat_trans, cat_features)
#     ]
# )

# pipeline = Pipeline(steps = [
#     ('preprocessor', preprocessor)
# ])

# # corr = train_df.corr()

# print(train_x.head())

"""
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    

criterion = RMSLELoss()

rmsle = criterion(pred, actual)
"""