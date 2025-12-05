import joblib

model = joblib.load('logisticRegression.pkl')
new=[6,148,72,35,12,33.6,0.627,50]
result=model.predict([new])
print(result)