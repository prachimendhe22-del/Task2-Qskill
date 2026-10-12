import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\Desktop\New folder\Qskill_task2\house_prices.csv")

print("House Price Dataset:")
print(df)

X = df[["Area"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.to_list())

print("\nPredicted Prices:")
print(y_pred)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

area = pd.DataFrame({"Area": [1100]})
predicted_price = model.predict(area)

print("\nHouse Area:", 1100, "sq. ft.")
print("Predicted House Price: ₹", round(predicted_price[0], 2))

plt.scatter(df["Area"], df["Price"], label="Actual Prices")

plt.plot(
    df["Area"],
    model.predict(df[["Area"]]),
    label="Prediction Line"
)

plt.title("House Price Prediction")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price (₹)")
plt.legend()
plt.tight_layout()

plt.savefig("house_price_prediction.png")
plt.show()