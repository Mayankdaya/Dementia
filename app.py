from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

#load model
model = joblib.load('random_forest_model.joblib')

print('Classes:', model.classes_)

def categorize_mmse(mmse):
    """
    Approximate the encoding based on known distribution from training data
    where MMSE 26-30 was mostly encoded as 2
    """
    mmse = float(mmse)
    if mmse >= 26:     # Normal (most common, encoded as 2)
        return 2
    elif mmse >= 20:   # Mild impairment
        return 1
    elif mmse >= 10:   # Moderate impairment
        return 0
    else:             # Severe impairment
        return 3

# You might want to print the encoding mapping to verify
#print("\nMMSE Encoding Map:")
#for score in range(0, 31):
#    print(f"MMSE Score {score} -> Category {mmse_encoder.transform([score])[0]}")

def categorize_cdr(cdr):
    # Binary classification: 0 for normal/questionable (CDR 0-0.5), 1 for dementia (CDR 0.5+)
    return 1 if cdr >= 0.5 else 0

def test_model():
    print("\nTesting model with sample cases...")
    
    import pandas as pd
    
    # Get the best estimator from GridSearchCV
    best_model = model.best_estimator_
    
    # Updated feature names to match training
    feature_names = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MMSE', 'CDR']
    importances = best_model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # Test case - Dementia
    test_case_1 = pd.DataFrame([[
        1, 92, 10, 1.000000, 1453, 0.684000, 1.208000, 3, 1
    ]], columns=feature_names)

    # Test case 2 - Likely Not Demented
    test_case_2 = pd.DataFrame([[
        0, 79, 14, 3, 1308, 0.706000, 1.341000, 2, 0
    ]], columns=feature_names)

    print("\nTest Case 1 (Expected: Demented)")
    print("Prediction:", model.predict(test_case_1))
    print("Probabilities:", model.predict_proba(test_case_1))
    
    print("\nTest Case 2 (Expected: Not Demented)")
    print("Prediction:", model.predict(test_case_2))
    print("Probabilities:", model.predict_proba(test_case_2))

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/1')
def page_one():
    return render_template('1.html')

@app.route('/index')
def page_two():
    return render_template('index.html')

@app.route('/3')
def page_three():
    return render_template('3.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Updated feature names to match training
        feature_names = ['M/F', 'Age', 'EDUC', 'SES', 'eTIV', 'nWBV', 'ASF', 'MMSE', 'CDR']
        
        # Categorize MMSE and CDR
        mmse_category = categorize_mmse(float(data['mmse']))
        cdr_category = categorize_cdr(float(data['cdr']))
        
        # Create input DataFrame for model
        input_data = pd.DataFrame([[
            data['gender'],
            data['age'],
            data['educ'],
            data['ses'],
            data['etiv'],
            data['nwbv'],
            data['asf'],
            mmse_category,
            cdr_category
        ]], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Since classes are reversed (0 = Demented, 1 = Not Demented)
        result = int(prediction[0])
        
        return jsonify({
            'prediction': result,  # 1 means demented, 0 means not demented
            'probabilities': probabilities[0].tolist(),
            'input_values': {
                'gender': 'Male' if data['gender'] == 1 else 'Female',
                'age': data['age'],
                'educ': data['educ'],
                'ses': data['ses'],
                'mmse': f"{data['mmse']} (category: {mmse_category})",
                'etiv': data['etiv'],
                'nwbv': data['nwbv'],
                'asf': data['asf'],
                'cdr': f"{data['cdr']} (category: {cdr_category})"
            }
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Full error details: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    test_model()  # Run test cases before starting the server
    app.run(debug=True)
