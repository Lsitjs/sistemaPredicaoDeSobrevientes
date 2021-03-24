from django.shortcuts import render
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    titanic_train = pd.read_csv(r'C:\Users\jac\PycharmProjects\PredictTitanic\predict\notebook\train.csv')
    titanic_test = pd.read_csv(r'C:\Users\jac\PycharmProjects\PredictTitanic\predict\notebook\test.csv')

    age_mean = titanic_train['Age'].mean()
    age_mean = math.floor(age_mean)

    age_mean_test = titanic_test['Age'].mean()
    age_mean_test = math.floor(age_mean_test)

    titanic_train.update(titanic_train['Age'].fillna(age_mean))
    titanic_test.update(titanic_test['Age'].fillna(age_mean_test))

    le = LabelEncoder()
    le.fit_transform(titanic_train['Sex'])
    le.fit_transform(titanic_test['Sex'])

    titanic_train['New_sex'] = le.fit_transform(titanic_train['Sex'])
    titanic_test['New_sex'] = le.fit_transform(titanic_test['Sex'])

    features = ['Pclass', 'Age', 'New_sex']

    modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    x = titanic_train[features]
    y = titanic_train['Survived']

    modelo.fit(x, y)

    val1= float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = str(request.GET['n3'])

    if val3 == 'male':
        val3=1
    else:
        val3=0

    prev_x_test = titanic_test[features]

    previsao = modelo.predict([[val1,val2,val3]])

    result1=""
    if previsao ==[1]:
        result1 = 'Você sobreviverá, mas o navio vai afundar'
    else:
        result1 = 'Você não sobreviverá ao naufrágio'

    return render(request, 'predict.html', {"result2":result1})