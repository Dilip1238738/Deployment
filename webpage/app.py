from flask import Flask, render_template, request
import joblib

model=joblib.load('logisticRegression.pkl')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact_Us')
def contact():
    return render_template('contact.html')

@app.route('/submit', methods=['post'])
def predict():
    a=eval(request.form.get('preg'))
    b=eval(request.form.get('plas'))
    c=eval(request.form.get('pres'))
    d=eval(request.form.get('skin'))
    e=eval(request.form.get('test'))
    f=eval(request.form.get('mass'))
    g=eval(request.form.get('pedi'))
    h=eval(request.form.get('age'))
    result=model.predict([[a,b,c,d,e,f,g,h]])
    if result[0]==1:
        return 'you are safe'
    else:
        return 'you have diabetes'
app.run(debug=True, port='4789')