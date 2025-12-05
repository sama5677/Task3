import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

df = pd.merge(exercise, calories, on="User_ID")

df = df.drop_duplicates()


label = LabelEncoder()
df["Gender"] = label.fit_transform(df["Gender"])  


X = df[["Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp"]]
y = df["Calories"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("MSE :", mse)


new_data = pd.DataFrame({
    "Age": [25],
    "Weight": [70],
    "Height": [175],
    "Duration": [60],
    "Heart_Rate": [130],
    "Body_Temp": [36.5]  
})

prediction = model.predict(new_data)
print("Predicted Calories Burned =", prediction[0])


