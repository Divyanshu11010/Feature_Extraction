import os
import pandas as pd
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Importing constants from the src/constants.py file
from src.constants import entity_unit_map, allowed_units

# Function to extract text from an image URL using Tesseract OCR
def extract_text(image_url):
    print(f"Extracting text from image: {image_url}")
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    text = pytesseract.image_to_string(img)
    print(f"Extracted text: {text.strip()}")
    return text

# Build a regex pattern dynamically from the allowed units in the constants file
def build_unit_regex():
    units_pattern = "|".join(re.escape(unit) for unit in allowed_units)
    return rf"(\d+(\.\d+)?)\s?({units_pattern})"

# Function to extract numeric values and units using dynamically built regex
def extract_values_and_units(text):
    unit_pattern = build_unit_regex()  # Build regex from allowed units
    matches = re.findall(unit_pattern, text, re.IGNORECASE)
    
    valid_matches = []
    for match in matches:
        value, _, unit = match
        valid_matches.append((value, unit.lower()))
    
    print(f"Extracted values and units: {valid_matches}")
    return valid_matches

# Function to preprocess entity_value and extract numeric part
def preprocess_entity_value(value):
    print(f"Preprocessing entity_value: {value}")
    match = re.match(r"(\d+(\.\d+)?)", value)
    if match:
        return float(match.group(1))
    print("Warning: Could not extract a numeric value from entity_value")
    return None

# Function to train the Random Forest model
def train_model(train_csv):
    print(f"Loading training data from: {train_csv}")
    df = pd.read_csv(train_csv)
    
    # Preprocess entity_value
    print("Preprocessing target values (entity_value)...")
    df['entity_value'] = df['entity_value'].apply(preprocess_entity_value)
    
    # Remove rows where 'entity_value' is None
    df = df.dropna(subset=['entity_value'])
    print(f"Training data size: {df.shape}")
    
    # Features and target extraction
    X = df[['group_id', 'entity_name']]
    y = df['entity_value']
    
    # Preprocessing with OneHotEncoder that handles unknown categories
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['group_id', 'entity_name'])]
    )

    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor())])

    # Train-test split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training model with Random Forest...")
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model

# Predictor function that uses the trained model to make predictions
def predictor(image_link, group_id, entity_name, model):
    # Text extraction from image
    text = extract_text(image_link)
    
    # Extract values and units
    values_and_units = extract_values_and_units(text)
    
    # Prepare features for prediction
    print(f"Preparing features for prediction: group_id={group_id}, entity_name={entity_name}")
    feature_df = pd.DataFrame({
        'group_id': [group_id],
        'entity_name': [entity_name]
    })
    
    # Predict using the trained model
    prediction = model.predict(feature_df)
    predicted_value = prediction[0]
    print(f"Predicted value: {predicted_value}")
    
    # Check if the extracted unit is valid for the given entity_name
    formatted_prediction = None
    
    if entity_name in entity_unit_map:
        allowed_units_for_entity = entity_unit_map[entity_name]
        
        if values_and_units:
            # Get the first extracted unit and ensure it's valid for this entity
            extracted_value, extracted_unit = values_and_units[0]
            if extracted_unit in allowed_units_for_entity:
                formatted_prediction = f"{predicted_value} {extracted_unit}"
            else:
                # If unit is invalid, return prediction without units for measurable entities like width
                formatted_prediction = f"{predicted_value} (invalid unit for {entity_name})"
        else:
            # No units extracted, provide a default for entities with measurable units
            formatted_prediction = f"{predicted_value} {list(allowed_units_for_entity)[0]}"
    else:
        # For entities that don't use units (e.g., categorical attributes), just return the value
        formatted_prediction = f"{predicted_value}"
    
    print(f"Formatted prediction: {formatted_prediction}")
    return formatted_prediction

# Main execution
if __name__ == "__main__":
    DATASET_FOLDER = '/content/drive/My Drive/Colab Notebooks/Feature_Extraction/dataset'  # Adjust this path to where your data is in Google Drive
    
    # Load the model
    print("Starting the program...")
    model = train_model(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Load test data
    print(f"Loading test data from: {os.path.join(DATASET_FOLDER, 'test.csv')}")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Number of test samples: {test.shape[0]}")
    
    # Add index column if not present
    if 'index' not in test.columns:
        test['index'] = test.index

    # Make predictions
    print("Starting prediction for test data...")
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model), axis=1)
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    print(f"Saving predictions to: {output_filename}")
    test[['index', 'prediction']].to_csv(output_filename, index=False)
    
    print("Program completed successfully!")
