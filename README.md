# 🧠 **Spam email Classification**

## 📋 **OVERVIEW**
This project involves building and evaluating a **K-Nearest Neighbors (KNN)** model for classification tasks. The notebook includes:

- 📊 **Data Preprocessing**
- 🏋️‍♂️ **Model Training**
- 🧪 **Performance Evaluation**
- 📈 **Visualization of Results**

## 🔍 **DESCRIPTION**
The main steps include data processing and data understanding, then performing **tf-idf** normalization into vector data, and then performing testing on many popular classification models.
**Decision Tree**, **XGBClassifier**, **RandomForest**, **AdaBoost**, **KNeighbors**, **GradientBoosting**, **Stacking**, **Voting**
### 💡 **KNN (K-Nearest Neighbors)**
 - Supervised learning algorithm used for classification and regression.
 - It predicts based on the majority vote of its k-nearest data points.
 - Simple but sensitive to noisy data and requires proper scaling.
### 💡 **Decision Tree**
 - Uses a tree-like model of decisions, splitting data based on features.
 - Easy to interpret, but can overfit if not pruned properly.
 - Works well with small to medium-sized datasets.
### 💡 **Random Forest**
 - Ensemble model that combines multiple Decision Trees.
 - Reduces overfitting by averaging the predictions of individual trees.
 - Robust, accurate, and effective for a wide range of data.
### 💡 **XGBoost (Extreme Gradient Boosting)**
 - Powerful ensemble model using boosting with Decision Trees.
 - Optimized for speed and performance, often used in competitions.
 - Needs careful tuning to avoid overfitting.
### 💡 **AdaBoost (Adaptive Boosting)**
 - Focuses on correcting errors made by previous models in the ensemble.
 - Adjusts the weights of misclassified samples in each iteration.
 - Effective but sensitive to noisy data.
### 💡 **Gradient Boosting**
 - Boosting algorithm that minimizes prediction error using Gradient Descent.
 - Builds trees sequentially, each correcting the errors of the previous.
 - High accuracy but can be computationally expensive.
### 💡 **Stacking Classifier**
 - Ensemble method that combines multiple models using a meta-learner.
 - Each base model's predictions are used as input features for the final model.
 - Can leverage the strengths of diverse models for improved accuracy.
### 💡 **Voting Classifier**
- Combines predictions from different models using majority voting.
- Supports "hard voting" (majority class) and "soft voting" (probability-based).
- Simple yet effective ensemble approach for boosting model performance.
Open the Jupyter Notebook and execute the cells sequentially.


## 🛠 **DEPENDENCIES**

This project requires the following Python packages:

- 🐍 **Python 3.x**
- 📒 **Jupyter Notebook**
- 🗃️ **Pandas** - Data manipulation
- 🔢 **NumPy** - Numerical operations
- 🖼️ **Matplotlib** - Basic plotting
- 📊 **Seaborn** - Enhanced visualizations
- 📚 **NLTK** - Natural Language Processing
- 📝 **Regex** - Text preprocessing
- 💬 **WordCloud** - Text visualization

### Machine Learning
- 🌳 **DecisionTree** - Tree-based model
- 🤖 **RandomForest** - Ensemble of trees
- 📈 **LogisticRegression** - Linear classifier
- 🏘️ **KNN** - Nearest neighbors
- 🔥 **XGBoost** - Gradient boosting
- ⚡ **AdaBoost** - Adaptive boosting
- 🚀 **GradientBoosting** - Sequential boosting
- 🗳️ **VotingClassifier** - Majority voting
- 🔗 **StackingClassifier** - Meta-ensemble

### Preprocessing
- 🔠 **TfidfVectorizer** - Text vectorization
- 🔢 **LabelEncoder** - Encoding labels
- 📏 **MinMaxScaler** - Feature scaling

### Evaluation
- 📊 **ConfusionMatrix** - Error analysis
- 📈 **ROC Curve** - Model performance
- 📋 **Classification Report** - Metrics summary


You can install them with:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## 📂 **FILES**

- **\`KNN_Model.ipynb\`**: Main notebook containing the code and analysis for the KNN model.

## 📝 **USAGE**

To use this project, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd KNN_Kaggle_Train
   ```

3. **Open the Jupyter Notebook**:

   ```bash
   jupyter notebook KNN_Model.ipynb
   ```

## 📊 **RESULTS**
The KNN model's performance is evaluated using metrics such as:
- **Accuracy**
- **Precision**
- **Confusion Matrix**

The results are visualized with plots, and a detailed discussion is included in the notebook.

## 📄 **LICENSE**
This project is licensed under the **MIT License** - see the LICENSE file for details.

## 🌐 **REFERENCES**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [K-Nearest Neighbors Algorithm Explained](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
