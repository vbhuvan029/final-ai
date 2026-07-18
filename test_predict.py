from ml_model.predict import predict_disease
print("Test 1:", predict_disease(['itching', 'sweating', 'back_pain', 'constipation', 'coma']))
print("Test 2:", predict_disease(['skin_rash', 'stomach_pain', 'headache']))
