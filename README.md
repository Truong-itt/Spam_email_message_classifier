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

## 📝 **USAGE**

To explore and use this project, follow the link below:

- 📈 **View and Run the Notebook on Kaggle**:

  [Spam Message Classification Notebook on Kaggle](https://www.kaggle.com/code/hduytrng/spam-message)

Simply click the link above to access the full Jupyter Notebook directly on Kaggle, where you can view the code, run it, and interact with the data.


## 📊 **RESULTS**
The KNN model's performance is evaluated using metrics such as:
- **Accuracy**
- **Precision**
- **Confusion Matrix**

The results are visualized with plots, and a detailed discussion is included in the notebook.

## 📄 **LICENSE**
This project is licensed under the **MIT License** - see the LICENSE file for details.

## 🌐 **REFERENCES**
## 🌐 **REFERENCES**
- [K-Nearest Neighbors Algorithm Explained](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Decision Tree Algorithm Explained](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Random Forest Algorithm Explained](https://en.wikipedia.org/wiki/Random_forest)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [AdaBoost Algorithm Explained](https://en.wikipedia.org/wiki/AdaBoost)
- [Gradient Boosting Algorithm Explained](https://en.wikipedia.org/wiki/Gradient_boosting)
- [Stacking Classifier in Scikit-Learn](https://scikit-learn.org/stable/modules/ensemble.html#stacking)
- [Voting Classifier in Scikit-Learn](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
- [Logistic Regression Algorithm Explained](https://en.wikipedia.org/wiki/Logistic_regression)
- [Confusion Matrix and Performance Metrics](https://en.wikipedia.org/wiki/Confusion_matrix)
- [TF-IDF Vectorization Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

