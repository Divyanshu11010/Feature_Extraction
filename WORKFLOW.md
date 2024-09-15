# Image Feature Extraction Hackathon - Roadmap

## Day 1: Image Downloading and Text Extraction

### 1. Set Up the Environment (1-2 hours)
- Install Python and required libraries:
    ```bash
    pip install pandas numpy scikit-learn opencv-python pillow pytesseract
    ```
- Install Tesseract OCR for text extraction.

### 2. Download Images (2 hours)
- Use `src/utils.py` to download images from URLs in `train.csv` and `test.csv`.

### 3. Text Extraction from Images (3-4 hours)
- Use **Tesseract OCR** to extract text from images.
    ```python
    import pytesseract
    from PIL import Image
    import requests
    from io import BytesIO

    def extract_text(image_url):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return pytesseract.image_to_string(img)
    ```

### 4. Regex for Extracting Values and Units (1-2 hours)
- Use **Regular Expressions** to extract numeric values and corresponding units.
    ```python
    import re

    text = extract_text(image_link)
    pattern = r"(\\d+(\\.\\d+)?)\\s?(gram|kg|cm|ml|ounce)"
    matches = re.findall(pattern, text)
    ```

---

## Day 2: Model Training, Prediction, and Validation

### 1. Simple ML Model Training (3-4 hours)
- Train a simple **Random Forest Regressor** on the features (`group_id`, `entity_name`) and target (`entity_value`) in `train.csv`.
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Assuming features are extracted using one-hot encoding for categorical variables
    X_train, X_test, y_train, y_test = train_test_split(features, target_values, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ```

- Use **one-hot encoding** to convert categorical features into numerical values.

### 2. Prediction on Test Set (2-3 hours)
- Extract features from the test set and use your trained model to predict the `entity_value`.
- Format the predictions correctly (e.g., "2 grams", "5.6 centimetres").

### 3. Validation (1 hour)
- Run the `sanity.py` script to ensure your output file is correctly formatted.

---

## Key Resources

1. **Tesseract OCR**: [Python-Tesseract GitHub](https://github.com/madmaze/pytesseract)
2. **Scikit-learn (Random Forest)**: [Scikit-learn Docs](https://scikit-learn.org/stable/)

"""