import numpy as np
import pandas as pd
import category_encoders as ce
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Using MAPE error metrics to check for the error rate and accuracy level
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

data = pd.read_csv('./dataset/DummySalaryData.csv')
print(data.head(5))
data = data.drop('ID', axis=1)
data1 = data.drop('Salary', axis=1)
c5=data1.columns
target_encoding=np.array(data['Salary'])

encoder5 = ce.TargetEncoder(cols=[x for x in c5])

target_encoder_all2= encoder5.fit_transform(data1,target_encoding)


#polydataset = target_encoder_all2.drop('Salary', axis=1)
polydataset = target_encoder_all2.copy()

poly = PolynomialFeatures(degree=5, include_bias=False)
polydataset = np.array(polydataset)

poly_features = poly.fit_transform(polydataset)

poly_reg_model = LinearRegression()

poly_reg_model.fit(poly_features, np.array(data['Salary']))
Salary_predicted = poly_reg_model.predict(poly_features)

lin_reg_rmse = np.sqrt(mean_squared_error(data['Salary'], Salary_predicted))

Actual = data['Salary']

X, y = poly_features, np.array(data['Salary'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=12)
Xpd = pd.DataFrame(poly_features)
Xpd.describe()

# Instance and fit
knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

# Save trained to file
joblib.dump(encoder5,"./trained/encoderfile.sav")
joblib.dump(poly,"./trained/polyfile.sav")
joblib.dump(knn_model,"./trained/trainedmodel.sav")