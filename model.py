import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

class HousePricePredictor:
    def __init__(self, csv_path: str):
        """Initialize the predictor by loading and processing data."""
        print("Loading data...")
        self.data = pd.read_csv(csv_path)
        print("Data loaded. Size: ", self.data.shape)
        
        self.clean_data()
        self.eda()  # Add EDA
        self.hypothesis_testing()  # Add Hypothesis Testing
        self.prepare_data()
        self.train_and_evaluate_model()

    def eda(self):
        """Perform Exploratory Data Analysis."""
        print("Performing EDA...")
        sns.histplot(self.data['price'], bins=50, kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.savefig('price_distribution_histplot.png')  # Save the plot
        # plt.show()
        
        sns.pairplot(self.data[['price', 'rooms', 'bathrooms', 'size']])
        # plt.show()
        plt.savefig('pairplot.png')  # Save the plot
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        # plt.show()
        plt.savefig('correlation_heatmap.png')  # Save the plot
        
    def hypothesis_testing(self):
        """Perform Hypothesis Testing."""
        print("Performing Hypothesis Testing...")
        condo_prices = self.data[self.data['property_type'] == 'condominium']['price']
        terrace_prices = self.data[self.data['property_type'] == 'terrace']['price']
        
        t_stat, p_value = stats.ttest_ind(condo_prices, terrace_prices)
        print(f"T-test between condominium and terrace prices: t-statistic = {t_stat}, p-value = {p_value}")
        
        if p_value < 0.05:
            print("Reject the null hypothesis: There is a significant difference between condominium and terrace prices.")
        else:
            print("Fail to reject the null hypothesis: No significant difference between condominium and terrace prices.")
        
    def clean_data(self):
        """Clean the dataset by removing unnecessary columns and handling missing values."""
        print("Cleaning data...")
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')  # Normalize column names
        string_columns = list(self.data.dtypes[self.data.dtypes == 'object'].index)
        for col in string_columns:
            self.data[col] = self.data[col].str.strip().str.lower().str.replace(' ', '_')
        
        # Clean 'price' and 'size' columns
        self.data['price'] = self.data['price'].str.replace(',', '').str.extract('(\d+)').astype(float)
        self.data['size'] = self.data['size'].str.replace(',', '').str.extract('(\d+)').astype(float)
        
        # Transform 'rooms' column
        self.data['rooms'] = self.data['rooms'].apply(lambda x: sum(map(int, x.split('+'))) if (pd.notnull(x) and all(part.isdigit() for part in x.split('+'))) else np.nan)
        
        # Drop rows with missing values in crucial columns
        self.data = self.data.dropna(subset=['size', 'price', 'rooms', 'property_type'])
        
        # Fill missing values in 'furnishing', 'car_parks', and 'bathrooms'
        self.data['furnishing'] = self.data['furnishing'].fillna('unknown')
        self.data['car_parks'] = self.data['car_parks'].fillna(self.data['car_parks'].mean()).astype(int)
        self.data['bathrooms'] = self.data['bathrooms'].fillna(self.data['bathrooms'].mean()).astype(int)
        
        # Group similar property types
        property_mapping = {
            'condominium': 'condominium',
            'terrace': 'terrace',
            'residential_land': 'terrace',
            'bungalow': 'bungalow',
            'semi-detached_house': 'semi-detached_house',
            'flat': 'flat',
            'townhouse': 'terrace',
            'apartment': 'apartment',
            'serviced_residence': 'serviced_residence',
            'cluster_house': 'terrace'
        }
        self.data['property_type'] = self.data['property_type'].apply(lambda x: property_mapping.get(x, x))
        
        print("Data Cleaning Complete")

    def prepare_data(self):
        """Prepare the data for model training by splitting into train and test sets and scaling features."""
        print("Preparing data for modeling...")
        print("Columns in data before processing:", self.data.columns.tolist())  # Print columns before processing

        # Separate target variable and features
        self.y = self.data['price']
        self.X = self.data.drop('price', axis=1)
        
        # Define preprocessing steps
        numeric_features = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Apply transformations
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        
        print("Data preparation complete. Sizes: ", self.X_train.shape, self.X_test.shape)
        # Construct feature names
        num_feature_names = numeric_features.tolist()
        cat_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features).tolist()
        self.feature_names_out = num_feature_names + cat_feature_names
        print("Feature names after preprocessing:", self.feature_names_out)

    def train_and_evaluate_model(self):
        """Train the Ridge regression model and evaluate its performance."""
        print("Training the model...")
        self.model = Ridge(alpha=1.0)
        self.model.fit(self.X_train, self.y_train)
        
        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
        print(f"Mean Squared Error: {self.mse}")
        print(f"R-squared: {self.r2}")

        # Cross-Validation Score
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")
    
    def predict_price(self, features: list) -> float:
        """Predict house price for the given feature set."""
        # Construct a dictionary with the feature names and values
        feature_dict = dict(zip(['rooms', 'bathrooms', 'size', 'car_parks', 'location', 'property_type', 'furnishing'], features))
        
        # Convert the dictionary to a DataFrame
        features_df = pd.DataFrame([feature_dict])
        
        # Apply the same preprocessing steps as training data
        features_transformed = self.preprocessor.transform(features_df)
        
        return self.model.predict(features_transformed)[0]

# Example usage
if __name__ == "__main__":
    predictor = HousePricePredictor('mas_housing.csv')

    # Define a base feature set (update with appropriate feature values as needed)
    base_features = [
        3.0,  # Rooms
        1.5,  # Bathrooms
        1340,  # Size
        0,  # Car Parks
        'klcc,_kuala_lumpur',  # Location
        'condominium',  # Property Type
        'fully_furnished'  # Furnishing
    ]

    # Example prediction
    predicted_price = predictor.predict_price(base_features)
    print(f"Predicted Price: RM {predicted_price:.2f}")
