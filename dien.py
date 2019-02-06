import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns
sns.set_style('ticks')

import itertools
%matplotlib inline

df = pd.read_csv("breast-cancer-wisconsin.data",header = None )

df.columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 
              'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']

df.head(3)
#checking datatype of each features in data set
df.dtypes
df.nuclei.unique()
df.nuclei = pd.to_numeric(df.nuclei, errors ='coerce')
#I want add the label columns to show the type of status for visulation only, no need for modeling
def set_status_class(x):
    return ['benign' if (y == 2) else 'malignant' for y in x]

df['label'] = set_status_class(df['status'])
# checking the NaN values in data set
df.isnull().any()
# replace the NaN values with mean value for each features and drop the "sample". It is not useful for modeling 
df= df.fillna(df.mean())
df = df.drop(['sample'], axis =1)

# Get the list of columns in dataframe, we can use it for normalization later
col_list = df.columns
# Quick look on statistic information from data 
df.describe()

# I will do normalization on the numerical columns from the data frame 
# Using the MinMax Scaler

norm_list = ['thickness', 'size', 'shape', 'adhesion', 'epithelial',
       'nuclei', 'chromatin', 'nucleoli', 'mitoses','status']

minmax = MinMaxScaler()
df[norm_list] = minmax.fit_transform(df[norm_list])

print("Dataframe shape: ", df.shape)
df.head(3)

#checking the number of patient in two classes
sns.countplot(x = "label", hue ="label", data =df)
# The box plots for ALL numerical columns

norm_list = ['thickness', 'size', 'shape', 'adhesion', 'epithelial',
       'nuclei', 'chromatin', 'nucleoli', 'mitoses']

fig,axes = plt.subplots(3,3, figsize = (12,6))
for i, t in enumerate(norm_list):
    sns.boxplot(y =t, x = "label", data= df, ax = axes[i//3, i%3]) 
    
fig.savefig("boxplot_numerical.pdf")

# try with violin plots:
fig,axes = plt.subplots(3,3, figsize = (15,12))
for i, t in enumerate(norm_list):
    sns.violinplot(y =t, x = "label", data= df, ax = axes[i//3, i%3]) 
    
fig.savefig("violin_numerical.pdf")

# The histogram plots for ALL numerical columns
fig,axes = plt.subplots(3,3, figsize = (15,9))
for i, t in enumerate(norm_list):
    sns.countplot(x =t, hue = "label", data= df, ax = axes[i//3, i%3])
# I want to try wise pair plots to see the correlation 
list_check = ['thickness', 'size', 'shape','nuclei']
sns.pairplot(df, x_vars = list_check, y_vars = list_check,
             hue = 'label', size =3)
# Checking the correlation between different faetures
df.corr()

# using the imshow to plot the correlation matrix
plt.figure(figsize= (6,6))
plt.imshow(df[norm_list].corr(), cmap = plt.cm.Reds, interpolation = 'nearest')
plt.colorbar()

tick_marks = [i for i in range(len(df[norm_list].columns))]
plt.xticks(tick_marks, df[norm_list].columns, rotation = 'vertical')
plt.yticks(tick_marks, df[norm_list].columns)
plt.show()

# setting label in to data frame y and data set to datafram x
y = df['status']
X = df.drop(["status", 'label'], axis = 1)

# sliting 50% of data to train set and 50% to the test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 7)

# Checking the balance of data  in Train and Test set
print("0: benign , 1: malignant")
print("For train data: ")
print(pd.value_counts(y_train, normalize = True))
print("For test data:  ")
print(pd.value_counts(y_test, normalize = True))

# try with linear_regression model 
linear = LogisticRegression()
linear.fit(x_train, y_train)

linear.score(x_test, y_test)
y_linear = linear.predict(x_test)
confusion_matrix(y_test, y_linear)
class_names = ['benign', 'malignant']
print(classification_report(y_test, y_linear, target_names = class_names))
# try randome forest 
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc.score(x_test,y_test)
y_rfc = rfc.predict(x_test)
confusion_matrix(y_test, y_rfc)
print(classification_report(y_test, y_rfc, target_names = class_names))

# Create the adaboost model and use the gridsearchCV to get the best parameters

ada_clf = AdaBoostClassifier()

param_grid ={
        'n_estimators': [100, 200,500],
        'learning_rate': [0.2,0.5,1.0],
},
grid_ada = GridSearchCV(ada_clf, cv=3, n_jobs=3, param_grid=param_grid)

grid_ada.fit(x_train, y_train)
# checking the best parameters from gridsearchCV for adaboosted classifier
grid_ada.best_params_
# using the model above to do prediction on test data
y_ada = grid_ada.predict(x_test)
# Checking the accuracy of model 
print("The accurary of AdaBoosted Classification model: ", grid_ada.score(x_test, y_test))
# Checking the confusion Matrix
class_names = ['benign', 'malignant']
cnf_mat = confusion_matrix(y_test, y_ada)
cnf_mat
# Function to do confusion matrix plot for better visualization 
def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(cnf_mat, classes=class_names, title='Confusion matrix')
#Checking the summary of classifier performance
print(classification_report(y_test, y_ada, target_names = class_names))
# Checking the True postitive and Fall positive rate for ROC curver visualization
y_grid_ada_score = grid_ada.decision_function(x_test)
fpr_grid_ada, tpr_grid_ada, thresholds_grid_ada = roc_curve(y_test, y_grid_ada_score)
#plot the ROC curve  
plt.figure(figsize=(6,5))
plt.plot(fpr_grid_ada, tpr_grid_ada, label='Adaboost with the best Pars')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='random', alpha=.8)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.grid()
plt.legend()
plt.axes().set_aspect('equal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# I will use isomap to compress data to 2 dimentions features
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors = 5, n_components =2)
iso.fit(x_train)
x_train = iso.transform(x_train)
x_test = iso.transform(x_test)

print(" Shape of train data: ", x_train.shape)
print(" Shape of test data : ", x_test.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5, weights = 'distance')
knn.fit(x_train, y_train)
# .. your code changes above ..
knn.score(x_test, y_test)
y_knn = knn.predict(x_test)
# Checking the confusion Matrix
cnf_mat_knn= confusion_matrix(y_test, y_knn)
plot_confusion_matrix(cnf_mat_knn, classes=class_names, title='Confusion matrix')
#Checking the summary of classification
print(classification_report(y_test, y_knn, target_names = ['benign', 'malignant']))
# plot the boundary condition for knn
def plotDecisionBoundary(model, X, y):
    print("Color: Blue: benign")
    print('Salmom: maglinant')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    padding = 0.1
    resolution = 0.1

    #(0 for benign, 1 for malignant)
    colors = {0:'royalblue', 1:'lightsalmon'} 


    # Calculate the boundaris
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
    plt.axis('tight')

    ll = {0: 'Benign', 1: "Malignant"}
    #Plot your testing points as well...
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8, label=ll[label])

    p = model.get_params()
    plt.title('K = ' + str(p['n_neighbors']))
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large',frameon = True)
    legend.get_frame().set_facecolor('white')
    fig.savefig("Decision_boundary.pdf")
    plt.show()

plotDecisionBoundary(knn, x_test, y_test)
#fig.savefig("Decision_boundary.pdf")
