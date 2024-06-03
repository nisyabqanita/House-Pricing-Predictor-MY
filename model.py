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
import joblib

class HousePricePredictor:
    def __init__(self, csv_path: str):
        """Initialize the predictor by loading and processing data."""
        print("Loading data...")
        self.data = pd.read_csv(csv_path)
        print("Data loaded. Size: ", self.data.shape)
        self.clean_data()
        self.eda()
        self.hypothesis_testing()
        self.prepare_data()
        self.train_and_evaluate_model()
        self.save_model()

    def eda(self):
        """Perform Exploratory Data Analysis."""
        print("Performing EDA...")
        sns.histplot(self.data["price"], bins=50, kde=True)
        plt.title("Price Distribution")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.savefig("price_distribution_histplot.png")
        plt.clf()

        sns.pairplot(self.data[["price", "rooms", "bathrooms", "size"]])
        plt.savefig("pairplot.png")
        plt.clf()

        plt.figure(figsize=(10, 8))
        numeric_data = self.data.select_dtypes(include=["number"])
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.clf()

    def hypothesis_testing(self):
        """Perform Hypothesis Testing."""
        print("Performing Hypothesis Testing...")
        condo_prices = self.data[self.data["property_type"] == "condominium"]["price"]
        terrace_prices = self.data[self.data["property_type"] == "terrace"]["price"]
        t_stat, p_value = stats.ttest_ind(condo_prices, terrace_prices)
        print(f"T-test between condominium and terrace prices: t-statistic = {t_stat}, p-value = {p_value}")
        if p_value < 0.05:
            print("Reject the null hypothesis: There is a significant difference between condominium and terrace prices.")
        else:
            print("Fail to reject the null hypothesis: No significant difference between condominium and terrace prices.")

    def clean_data(self):
        """Clean the dataset by removing unnecessary columns and handling missing values."""
        print("Cleaning data...")
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(" ", "_")
        string_columns = list(self.data.dtypes[self.data.dtypes == "object"].index)
        for col in string_columns:
            self.data[col] = self.data[col].str.strip().str.lower().str.replace(" ", "_")
        self.data["price"] = self.data["price"].str.replace(",", "").str.extract(r"(\d+)").astype(float)
        self.data["size"] = self.data["size"].str.replace(",", "").str.extract(r"(\d+)").astype(float)
        self.data["rooms"] = self.data["rooms"].apply(lambda x: sum(map(int, x.split("+"))) if (pd.notnull(x) and all(part.isdigit() for part in x.split("+"))) else np.nan)
        self.data = self.data.dropna(subset=["size", "price", "rooms", "property_type"])
        self.data["furnishing"] = self.data["furnishing"].fillna("unknown")
        self.data["car_parks"] = self.data["car_parks"].fillna(self.data["car_parks"].mean()).astype(int)
        self.data["bathrooms"] = self.data["bathrooms"].fillna(self.data["bathrooms"].mean()).astype(int)
        property_mapping = {
            "condominium": "condominium",
            "terrace": "terrace",
            "residential_land": "terrace",
            "bungalow": "bungalow",
            "semi-detached_house": "semi-detached_house",
            "flat": "flat",
            "townhouse": "terrace",
            "apartment": "apartment",
            "serviced_residence": "serviced_residence",
            "cluster_house": "terrace",
        }
        self.data["property_type"] = self.data["property_type"].apply(lambda x: property_mapping.get(x, x))
        print("Data Cleaning Complete")

    def prepare_data(self):
        """Prepare the data for model training by splitting into train and test sets and scaling features."""
        print("Preparing data for modeling...")
        print("Columns in data before processing:", self.data.columns.tolist())
        self.y = self.data["price"]
        self.X = self.data.drop("price", axis=1)
        numeric_features = self.X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = self.X.select_dtypes(include=["object"]).columns
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        print("Data preparation complete. Sizes: ", self.X_train.shape, self.X_test.shape)
        num_feature_names = numeric_features.tolist()
        cat_feature_names = self.preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features).tolist()
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
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

    def save_model(self):
        """Save the trained model and preprocessor to a file."""
        print("Saving the model...")
        joblib.dump(self.model, "house_price_predictor_model.sav")
        joblib.dump(self.preprocessor, "preprocessor.sav")
        print("Model and preprocessor saved.")

    def predict_price(self, features: list) -> float:
        """Predict house price for the given feature set."""
        feature_dict = dict(
            zip(
                ["rooms", "bathrooms", "size", "car_parks", "location", "property_type", "furnishing"],
                features,
            )
        )
        features_df = pd.DataFrame([feature_dict])
        features_transformed = self.preprocessor.transform(features_df)
        return self.model.predict(features_transformed)[0]

# Example usage
if __name__ == "__main__":
    predictor = HousePricePredictor("mas_housing.csv")
    base_features = [
        3.0,  # Rooms
        1.5,  # Bathrooms
        1340,  # Size
        0,  # Car Parks
        "klcc,_kuala_lumpur",  # Location
        "condominium",  # Property Type
        "fully_furnished",  # Furnishing
    ]
    predicted_price = predictor.predict_price(base_features)
    print(f"Predicted Price: RM {predicted_price:.2f}")
