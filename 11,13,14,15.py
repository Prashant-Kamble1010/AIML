import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Example dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000,1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4,2, 3, 3, 4, 4],
    'location': [1, 2, 2,1, 2, 2, 3, 3, 3, 3],  
    'price': [200000, 250000,200000, 250000, 300000, 350000, 40000, 300000, 350000, 400000]
}

df = pd.DataFrame(data)

# Features & target
X = df[['area', 'bedrooms', 'location']]
y = df['price']

# Model
model = LinearRegression()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("R² Scores from K-Fold:", scores)
print("Mean R²:", scores.mean())
