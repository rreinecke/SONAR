from sklearn.datasets import load_iris
import pandas as pd
from build_tree import SONAR

"""
The classical Iris dataset is too small for SONAR.
This only demonstrates the use of SONAR with any dataset in a small example.
"""

iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame({iris.feature_names[0]: X[:, 0], iris.feature_names[1]: X[:, 1], iris.feature_names[2]: X[:, 2],
                   iris.feature_names[3]: X[:, 3]})

inputs = iris.feature_names
sonar = SONAR(df, inputs, categoricals=iris.target_names)
sonar.prepare("petal length (cm)")
sonar.tree()
