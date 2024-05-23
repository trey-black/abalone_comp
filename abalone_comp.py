import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss

###################### MODEL PARAMS ######################

num_epochs = 500
batch_size = 4
learning_rate = 0.001
momentum = 0.95

###################### DATA IMPORT ######################

data = pd.read_csv('/Users/treyb/Documents/Data Science/Programs/abalone_comp/train.csv', index_col=0)
x = data.drop(['Rings'], axis=1)
y = data['Rings'].astype('object')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 888)

###################### BASELINE SUCCESS RATE ######################

###################### VISUAL INSPECTION ######################

# train.hist()
# plt.show()
# stats.probplot(dist, dist='norm', plot=plt)
# plt.show()

num_features = x_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = x_train.select_dtypes(include=['object']).columns

# print(x_train.columns)

###################### NORMALITY ASSESSMENT ######################

# need to look into the error for shapiro w/ n > 5000

# Create a matrix to evaluate normality of data
normality_cols = [['col', 'skewness', 'kurtosis', 'shap stat', 'shap p val']]
normality_tests = []
for col in x_train[num_features]:
    dist = x_train[col].values
    skewness = stats.skew(dist)
    kurt = stats.kurtosis(dist)
    stat, p_value = stats.shapiro(dist)
    normality_tests.append([col, skewness, kurt, stat, p_value])

# print(pd.DataFrame(normality_tests, columns=normality_cols))

# Length, Diameter, Whole weight.1 are moderately skewed
# Height has a high kurtosis value (notable tailedness)

###################### EDA ######################

# print(np.sort(pd.unique(y_train)))
# print(y_train.value_counts())
# y has 28 categories but the range goes from 1-29 skipping 28
# probably okay since 0.17% of abalones were older than 24

# Note sex having a third category

# train_df.info()
# No nulls in the data, sex needs to be encoded, numerical data needs to be standardized

###################### CORRELATION MATRIX ######################

# # corr = train_df.corr()

###################### PROCESSING PIPELINE ######################

num_trans = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_trans = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers = [
        ('numeric', num_trans, num_features),
        ('categorical', cat_trans, cat_features)
    ]
)

pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor)
])

x_train = pipeline.fit_transform(x_train)
x_test = pipeline.transform(x_test)

###################### DATALOADER ######################

# Should I convert y to a categorical variable?
# I guess I could run this as both a regression problem and a categorical problem, so might be worth setting up infrastructure for both?
# Let's start with categorical and then run everything for numerical

train_features = torch.from_numpy(x_train)
train_labels = torch.from_numpy(y_train.to_numpy(dtype='float64'))

val_features = torch.from_numpy(x_test)
val_labels = torch.from_numpy(y_test.to_numpy(dtype='float64'))

train_dataset = TensorDataset(train_features.float(), train_labels.float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_features.float(), val_labels.float())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# test_features, test_labels = next(iter(train_loader))
# print(f'Feature batch example: {test_features.size()}')
# print(f'Label batch example: {test_labels.size()}')

###################### MODEL DEFINITION ######################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8,32)
        self.fc2 = nn.Linear(32,28)
        # self.dropout1 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(x, dim=-1)
        # x = self.dropout1(x)
        return x

###################### ACCURACY FUNCTION DEFINITION ######################

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))    

criterion = RMSLELoss()

# rmsle = criterion(pred, actual)

###################### TRAINING ######################

###################### EVALUATION ######################

###################### CUSTOM LOSS ######################

###################### HYPERPARAMETER TUNING ######################

