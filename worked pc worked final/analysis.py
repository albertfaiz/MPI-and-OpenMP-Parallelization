import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("TensorFlow version:", tf.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Simple NumPy operation
a = np.array([1, 2, 3])
print("NumPy array:", a * 2)

# Simple Pandas DataFrame
data = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data)
print("Pandas DataFrame:\n", df)

# Simple Matplotlib plot (won't display in terminal, but should import without error)
plt.plot([1, 2], [3, 4])
plt.title("Simple Plot")
# plt.show() # Uncomment if you want to display in a graphical environment

print("All essential packages imported successfully!")