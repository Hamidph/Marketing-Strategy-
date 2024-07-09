## Machine Learning: Support Vector Machine

# In this section, we will use the Support Vector Machine (SVM) algorithm to build a model to predict customer churn.

# Load the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Load the data
df = pd.read_csv('cleaned_dataset.csv')

# Create the feature matrix and target variable
X = df.drop("Churn", axis=1) # Features
y = df["Churn"] # Target variable

from sklearn.preprocessing import OneHotEncoder

#  Define the columns to be encoded
columns_to_encode = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Select the columns to be encoded from the DataFrame
X_to_encode = X[columns_to_encode]

# Initialize the OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)  # Use sparse=False to get a dense matrix

# Fit and transform the selected columns
encoded_columns = cat_encoder.fit_transform(X_to_encode)

# Create new column names for the encoded values
new_columns = cat_encoder.get_feature_names_out(columns_to_encode)

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_columns, columns=new_columns)

# Drop the original columns from X
X = X.drop(columns=columns_to_encode)

# Concatenate the original DataFrame with the encoded DataFrame
X = pd.concat([X, encoded_df], axis=1)


# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Creating the SVM model
model = SVC(kernel='rbf')

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Here we will evaluate the model using the accuracy score and F1 score

# Model accuracy

# Calculate the accuracy of the model

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy:', accuracy)

y_test = y_test.map({'No': 0, 'Yes': 1})
y_pred = np.where(y_pred=='No', 0, 1)

# Now calculate the scores
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy:', accuracy)

precision = precision_score(y_test, y_pred)
print('Model precision:', precision)

recall = recall_score(y_test, y_pred)
print('Model recall:', recall)

f1 = f1_score(y_test, y_pred)
print('Model F1 score:', f1_score)

# Model precision
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Model precision:', precision)

# Model recall
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print('Model recall:', recall)

# Model F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print('Model F1 score:', f1)


# Model precision for class 1 which is 'Yes'
precision = precision_score(y_test, y_pred, pos_label=1)
print('Model precision for class 1:', precision)

# Model recall for class 1 which is 'Yes'
recall = recall_score(y_test, y_pred, pos_label=1)
print('Model recall for class 1:', recall)
      
# Model F1 score for class 1 which is 'Yes'
f1 = f1_score(y_test, y_pred, pos_label=1)
print('Model F1 score for class 1:', f1)

# Model precision for class 0 which is 'No'
precision = precision_score(y_test, y_pred, pos_label=0)
print('Model precision for class 0:', precision)

# Model recall for class 0 which is 'No'
recall = recall_score(y_test, y_pred, pos_label=0)

print('Model recall for class 0:', recall)
# Model F1 score for class 0 which is 'No'
f1 = f1_score(y_test, y_pred, pos_label=0)

# plot the ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# calculate the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# calculate the classification report
from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred)
print(report)

# why I cant see the heatmap of the confusion matrix
# I think the problem is with the values of y_pred, they are not binary
# I will try to convert them to binary values
# Convert probabilities to class labels (assuming threshold of 0.5)
y_pred = np.where(y_pred > 0.5, 1, 0)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# what is class 1?
# class 1 is the positive class which is 'Yes' in our case

# what is class 0?
# class 0 is the negative class which is 'No' in our case

# what is the precision for class 1?
# in churn data

# f1 score for class 0
f1 = f1_score(y_test, y_pred, pos_label=0)
print('Model F1 score for class 0:', f1)

