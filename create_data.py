import pandas as pd
import numpy as np

# Generate the synthetic dataset
np.random.seed(100)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
data = pd.DataFrame(np.hstack((X, y)), columns=['X', 'y'])

# Save the dataset
data.to_csv('data.csv', index=False)
print("Dataset created and saved to data.csv")