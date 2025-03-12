from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: model.pkl not found. Make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Preprocessing Mappings (as per your code)
owner_mapping = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 5
}
fuel_mapping = {
    'Diesel': 1,
    'Petrol': 2,
    'LPG': 3,
    'CNG': 4
}
seller_type_mapping = {
    'Individual': 1,
    'Dealer': 2,
    'Trustmark Dealer': 3
}
transmission_mapping = {
    'Manual': 1,
    'Automatic': 2
}
name_mapping = {
    'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
    'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
    'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
    'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 'Force': 27,
    'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31
}

# Cleaning function for mileage, engine, max_power
def clean_data(value):
    value_str = str(value)  # Ensure value is treated as a string
    value_split = value_str.split(' ')[0]
    value_stripped = value_split.strip()
    if value_stripped == '':
        return 0.0  # Return float 0.0 to match model expectation
    try:
        return float(value_stripped)
    except ValueError:
        return 0.0 # Handle cases where conversion to float fails, return 0.0


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            year = int(request.form['year'])
            km_driven = int(request.form['km_driven'])
            fuel = request.form['fuel']
            seller_type = request.form['seller_type']
            transmission = request.form['transmission']
            owner = request.form['owner']
            mileage_str = request.form['mileage'] # Get as string from form
            engine_str = request.form['engine']     # Get as string from form
            max_power_str = request.form['max_power'] # Get as string from form
            seats = int(request.form['seats'])

            # Preprocess categorical features using the mappings
            name_processed = name_mapping.get(name, 0) # Default to 0 if name not found (handle unknown names)
            fuel_processed = fuel_mapping.get(fuel, 0) # Default to 0 for unknown fuel
            seller_type_processed = seller_type_mapping.get(seller_type, 0) # Default to 0
            transmission_processed = transmission_mapping.get(transmission, 0) # Default to 0
            owner_processed = owner_mapping.get(owner, 0) # Default to 0

            # Clean and convert numerical features using the clean_data function
            mileage_processed = clean_data(mileage_str)
            engine_processed = clean_data(engine_str)
            max_power_processed = clean_data(max_power_str)


            # Create a DataFrame from the preprocessed input data
            input_data = pd.DataFrame({
                'name': [name_processed], # Use processed name
                'year': [year],# Review if this should be input or predicted
                'km_driven': [km_driven],
                'fuel': [fuel_processed], # Use processed fuel
                'seller_type': [seller_type_processed], # Use processed seller_type
                'transmission': [transmission_processed], # Use processed transmission
                'owner': [owner_processed], # Use processed owner
                'mileage': [mileage_processed], # Use processed mileage
                'engine': [engine_processed], # Use processed engine
                'max_power': [max_power_processed], # Use processed max_power
                'seats': [seats]
            })

            # Make prediction
            prediction = model.predict(input_data)
            output = round(prediction[0], 2) # Round to 2 decimal places

            return render_template('result.html', prediction_text='Predicted Car Price: ₹ {}'.format(output))

        except Exception as e:
            return render_template('index.html', error_message=f"Error processing input: {e}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True) # Set debug=False for production