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
import torch.optim as optim
import torchmetrics

###################### MODEL PARAMS ######################

num_epochs = 500
batch_size = 4

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

###################### MODEL DEFINITION ######################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8,32)
        self.fc2 = nn.Linear(32,29)
        # self.dropout1 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = F.softmax(x, dim=-1)
        # x = self.dropout1(x)
        return x

###################### CUSTOM LOSS DEFINITION ######################

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.epsilon = 1e-6  # Small constant to ensure numerical stability

    def forward(self, pred, actual):
        # Ensure that predictions and actual values are non-negative
        pred = torch.clamp(pred, min=0)
        actual = torch.clamp(actual, min=0)

        # Calculate the logarithms
        log_pred = torch.log(pred + 1 + self.epsilon)
        log_actual = torch.log(actual + 1 + self.epsilon)

        # Compute the mean squared error of the logarithms
        mse_loss = self.mse(log_pred, log_actual)
        
        # Return the square root of the MSE loss
        return torch.sqrt(mse_loss)   

criterion = RMSLELoss()

# rmsle = criterion(pred, actual)

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

###################### TRAINING LOOP FUNCTION ######################

# Accuracy metric:
metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)

def model_run(model, optimizer):
    eps = np.arange(1, num_epochs+1)
    g_train_loss = []
    g_val_loss = []
    model_accuracy = []
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        epoch_loss_train = 0.0
        epoch_loss_val = 0.0

        for batch_idx, data in enumerate(train_loader, 0):
            # data.requires_grad_() # ensure each batch of data requires gradients
            optimizer.zero_grad() # resets the gradients from the previous iteration
            features, targets = data
            features.requires_grad_()
            if 0 in targets: print(targets)
            # print(features)
            # print(targets.to(int))
            pred = model(features) # forward pass
            targets = F.one_hot(targets.to(int)-1, num_classes=29) # subtract one to adjust one hot indices
            # print(targets)
            # print(pred)
            # print(targets)
            loss = criterion(pred.double(), targets.double()) # loss computation
            loss.backward() # backward pass (calculates new gradients)
            optimizer.step() # update parameters
            train_loss += loss.item()
        epoch_loss_train = train_loss / len(train_loader)
        g_train_loss.append(epoch_loss_train)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                features, target = data
                pred = torch.argmax(model(features), dim=1)
                loss = criterion(pred, target)
                val_loss += loss.item()
                acc = metric(pred, target.argmax(dim=-1)) 
        epoch_loss_val = val_loss / len(val_loader)
        g_val_loss.append(epoch_loss_val)
        # print(f'Epoch loss: {epoch_loss}')
        acc = metric.compute()
        model_accuracy.append(acc.item())
        # print(f'Accuracy on all data: {acc}')
        metric.reset()
        model.train()
    return eps, g_train_loss, g_val_loss, model_accuracy

###################### TRAINING ######################

my_nn = Net()

optimizer_init = optim.Adam(my_nn.parameters())

eps, g_train_loss, g_val_loss, model_accuracy = model_run(my_nn, optimizer_init)

fig, ax = plt.subplots()
ax.plot(eps, g_train_loss, label='Training loss')
ax.plot(eps, g_val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.suptitle(f'Model accuracy: {round(max(model_accuracy), 4)*100}%')
plt.legend(loc="upper right")
plt.show()

# print(max(model_accuracy))

###################### EVALUATION ######################

# confusion matrix
# accuracy, F1, AUC, recall, and precision

###################### HYPERPARAMETER TUNING ######################

# learning curve

###################### PREDICTION ######################