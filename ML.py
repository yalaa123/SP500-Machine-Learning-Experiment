import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

sp = pd.read_csv("SP500.csv", na_values=".")
sp = sp.dropna()
ret = sp["SP500"].pct_change()
m1 = pd.DataFrame({"ret": ret})
m1["lag1"] = m1["ret"].shift(1)
m1["lag2"] = m1["ret"].shift(2)
m1["lag3"] = m1["ret"].shift(3)
m1 = m1.dropna()
n = int(len(m1) * 0.8)
train = m1.iloc[:n]
test = m1.iloc[n:]
features = ["lag1", "lag2", "lag3"]
model = LinearRegression()
model.fit(train[features], train["ret"])
print(model.score(test[features], test["ret"]))  ## R^2 score
pred = model.predict(test[features])
print((np.sign(pred) == np.sign(test["ret"])).mean()) ##Check directional accuracy
print(model.coef_)
##---------Predict tomorrows volatility
absret = ret.abs()
m2 = pd.DataFrame({"absret": absret})
m2["lag1"] = m2["absret"].shift(1)
m2["avg5"] = m2["absret"].rolling(5).mean().shift(1)
m2["avg20"] = m2["absret"].rolling(20).mean().shift(1)
m2 = m2.dropna()
n = int(len(m2) * 0.8)
train2 = m2.iloc[:n]
test2 = m2.iloc[n:]
features2 = ["lag1","avg5","avg20"]
model2 = LinearRegression()
model2.fit(train2[features2], train2["absret"])
##print(model2.coef_)
pred2 = model2.predict(test2[features2]) ## Predicts next day value based off fit model equation
print(model2.score(test2[features2], test2["absret"])) ## R^2 score
print(pred2.mean(), test2["absret"].mean()) ## Average predicted vs actual
print(np.corrcoef(pred2, test2["absret"])[0, 1]) ## Correlation
##--Predict the next 5 day average volatility

m3 = pd.DataFrame({"absret": absret})
m3["lag1"] = m3["absret"].shift(1)
m3["avg5"] = m3["absret"].rolling(5).mean().shift(1)
m3["avg20"] = m3["absret"].rolling(20).mean().shift(1)
m3["target"] = m3["absret"].rolling(5).mean().shift(-4)
m3 = m3.dropna()
##pd.set_option('display.max_rows', None)
n = int(len(m3) * 0.8)
train3 = m3.iloc[:n-4]
test3 = m3.iloc[n:]
model3 = LinearRegression()
model3.fit(train3[features2], train3["target"])
pred3 = model3.predict(test3[features2])
print(pred3.mean(), test3["target"].mean())
print(model3.score(test3[features2], test3["target"]))##evalute linear regression model
print(model3.coef_)
from sklearn.metrics import r2_score
print(r2_score(test3["target"], test3["avg20"]))##without using linear regression just using 20 day avg
print(np.corrcoef(pred3, test3["target"])[0, 1])##correlation



