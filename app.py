from flask import Flask,request, url_for, redirect, render_template
import pickle
import joblib
import detector
import numpy as np

app = Flask(__name__)

model=joblib.load(open('model.pkl','rb'))

def home():
    return render_template('homepage.html')



@app.route('/')
def home2():
    return render_template('homepage.html')


# @app.route('/error')
# def error():
#     return render_template('error.html')


# @app.route('/aboutproject')
# def aboutproject():
#     return render_template('aboutproject.html')



# @app.route('/review')
# def review():
#     return render_template('review.html')


# @app.route('/sourcecode')
# def sourcecode():
#     return render_template('sourcecode.html')

# @app.route('/creator')
# def creator():
#     return render_template('creator.html')





@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    
    arr = np.array([[data1, data2, data3]])
    output = model.predict(arr).item()  # Convert NumPy type to standard Python int

    if output < 4:
        return render_template('prediction.html', p=str(output), q='No')
    elif 4 <= output < 6:
        return render_template('prediction.html', p=str(output), q='Low')
    elif 6 <= output < 8:
        return render_template('prediction.html', p=str(output), q='Moderate')
    elif 8 <= output < 9:
        return render_template('prediction.html', p=str(output), q='High')
    elif output >= 9:
        return render_template('prediction.html', p=str(output), q='Very High')
    else:
        return render_template('prediction.html', p='N.A.', q='Undefined')
    
if __name__ == "__main__":
    app.run(debug=True)






