#import required libraries
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
import itertools

"""==========================================================================================================="""

cache = {}

"""==========================================================================================================="""

app = Flask(__name__)

"""==========================================================================================================="""

#load the dataset
dataset = pd.read_csv("D:\DM\pima_indians_diabetes.csv");
print("--------------------------------------------------------------------")
print("Pima_Inidan_Diabetes Dataset loaded")
print("--------------------------------------------------------------------")

"""==========================================================================================================="""

#visualizing dataset
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.title("Box plot")
plt.savefig('D:/DM/static/img/box_plot1.jpg', dpi = 100)

dataset.plot(kind= 'box' , figsize = (20, 10))
plt.title("Box plot")
plt.savefig('D:/DM/static/img/box_plot2.jpg', dpi = 100)

dataset.plot.scatter(x = 'serum_insulin', y = 'Triceps_skin_fold_thickness')
plt.title("Scatter Plot : Variables : serum_insulin and Triceps_skin_fold_thickness")
plt.savefig('D:/DM/static/img/scatter_plot1.jpg', dpi = 100)

dataset.plot.scatter(x = 'glucose_tolerance_test', y = 'Diastolic_blood_pressure')
plt.title("Scatter Plot : Variables : glucose_tolerance_test and Diastolic_blood_pressure")
plt.savefig('D:/DM/static/img/scatter_plot2.jpg', dpi = 100)

"""==========================================================================================================="""

#preprocessing
dataset = dataset[dataset['Triceps_skin_fold_thickness'] < 80]
dataset = dataset[dataset['serum_insulin'] < 600]

dataset.loc[dataset['serum_insulin'] == 0, 'serum_insulin'] = dataset['serum_insulin'].mean()
dataset.loc[dataset['glucose_tolerance_test'] == 0, 'glucose_tolerance_test'] = dataset['glucose_tolerance_test'].mean()
dataset.loc[dataset['Diastolic_blood_pressure'] == 0, 'Diastolic_blood_pressure'] = dataset['Diastolic_blood_pressure'].mean()
dataset.loc[dataset['Triceps_skin_fold_thickness'] == 0, 'Triceps_skin_fold_thickness'] = dataset['Triceps_skin_fold_thickness'].mean()
dataset.loc[dataset['Body_mass_index'] == 0, 'Body_mass_index'] = dataset['Body_mass_index'].mean()

dataset.to_csv("cleaned_data.csv")
print("Pima_Inidan_Diabetes Dataset Cleaned")
print("--------------------------------------------------------------------")

"""==========================================================================================================="""

#normalization
from sklearn import preprocessing
x = dataset.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled)
print("Pima_Inidan_Diabetes Dataset Normalized")
dataset.to_csv("normalized.csv")#saved normalized data
print("--------------------------------------------------------------------")

"""==========================================================================================================="""

#split our dataset into its attributes and labels.
X = dataset.iloc[:, 0:-1] #contains  columns i.e attributes
y = dataset.iloc[:, -1] #contains labels

#Train and Test Split
#80% train and 20% is test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)  

"""==========================================================================================================="""

@app.route("/")
def index():
    return render_template('index.html')

"""==========================================================================================================="""

#K Nearest Neighbour Classfier
@app.route("/knn", methods=['POST', 'GET'])
def knn():
    global k
    k = int(request.form['k'])
    print("-------------------Executing KNN-------------------")
    #train and predict
    cache['model'] = KNeighborsClassifier(n_neighbors=k)
    cache['model'].fit(X_train, y_train)  #the model gets train using this 

    #make predictions
    y_pred = cache['model'].predict(X_test)  

    #compare the accuracy
    accuracy = cache['model'].score(X_test, y_test)
    print(classification_report(y_test, y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,title='Confusion matrix, without normalization')
    plt.savefig('D:/DM/static/img/KNN_ConfusionMatrix.jpg')
    return render_template('knn.html', accuracy = (accuracy*100))

"""================================================================================================="""
#Gaussian Naive Bayes Classifier

@app.route("/nb")
def nb():
    print("-------------------Executing Naive Bayes-------------------")
    global model
    cache['model'] = GaussianNB()
    cache['model'].fit(X, y)

    #Predict on test data
    y_pred = cache['model'].predict(X_test)
    print(y_pred)

    #compare the accuracy
    accuracy = cache['model'].score(X_test, y_test)
    print(classification_report(y_test, y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,title='Confusion matrix')
    plt.savefig('D:/DM/static/img/NaiveBayes_ConfusionMatrix.jpg')

    return render_template('NaiveBayes.html', accuracy = (accuracy*100))

    #return """Accuracy of naive bayes classification algorithm on pima indian diabetes dataset is - """ + str(accuracy * 100) + """ %"""

"""================================================================================================="""
#Decision Tree Classifier

@app.route("/dt")
def dt():
    print("-------------------Executing Decision Tree-------------------")
    cache['model'] = tree.DecisionTreeClassifier(criterion='gini')

    # Train the model using the training sets and check score
    cache['model'].fit(X, y)

    #Predict Output
    y_pred = cache['model'].predict(X_test)
    print(y_pred)

    #compare the accuracy
    accuracy = cache['model'].score(X_test, y_test)
    print(classification_report(y_test, y_pred))
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,title='Confusion matrix')
    plt.savefig('D:/DM/static/img/DecisionTree_ConfusionMatrix.jpg')

    return render_template('DecisionTree.html', accuracy = (accuracy*100))

"""============================================================================="""
#predict
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    p1 = float(request.form['a'])
    p2 = float(request.form['b'])
    p3 = float(request.form['c'])
    p4 = float(request.form['d'])
    p5 = float(request.form['e'])
    p6 = float(request.form['f'])
    p7 = float(request.form['g'])
    p8 = float(request.form['h'])    

    Xnew = [[p1/100, p2/100, p3/1000, p4/100, p5/1000, p6/100, p7/1000, p8*0.006]]
    print(Xnew)
    predicted = cache['model'].predict(Xnew)

    #return """Predicted : """ + str(predicted)
    if predicted == 0 :
        return render_template('Predict.html', predicted = "with no")
    else:
        return render_template('Predict.html', predicted = "with")

"""================================================================================================="""
#Plotting confusion matrix

def plot_confusion_matrix(cm, normalize = False, title = 'Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


"""============================================================================="""

#executing using Flask

if __name__ == "__main__":
    app.run()
