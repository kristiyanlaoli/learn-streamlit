## Web
import streamlit as st
## EDA
import pandas as pd
## Visualisasi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
## Machine Learning
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class Web:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        st.title("Judul Yang Saya Buat")
        st.subheader("Ini subheader yang saya buat")
        st.markdown("""
        #### Description
        + This is a example EDA """)

    def eda (self, data) -> None:
        st.header("Explonatory Data Analysis")
        if data is not None:
            df = pd.read_csv(data)
            st.write(df)

            if st.checkbox("Show Shape"):
                st.write(df.shape)
            
            if st.checkbox("Show Columns"):
                all_columns =  df.columns.to_list()
                st.write(all_columns)

            if st.checkbox("Show Data Null"):
                st.write(df.isnull().sum())
            
            if st.checkbox("Show Duplicate Data"):
                st.write(df[df.duplicated()])
            
            if st.checkbox("Show Description Data"):
                st.write(df.describe())
            
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:-1].value_counts())

            if st.checkbox("Class Counts Bar Plot"):
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.title("Class Count Plot")
                st.write(sns.countplot(x=df["Species"]))
                st.pyplot()

            if st.checkbox("Distribution Bar Plot"):
                all_columns = df.columns.to_list()
                column1 = st.selectbox("Select X Column", all_columns)
                column2 = st.selectbox("Select Y Column", all_columns)
                plt.title("Distribution Species")
                st.write(sns.boxenplot(y=column2, x=column1, data=df, orient='v'))
                st.pyplot()

            if st.checkbox("Pair Plot"):
                st.write(sns.pairplot(df, hue="Species"))
                st.pyplot()

            if st.checkbox("Correlation Plot"):
                st.write(sns.heatmap(df.corr(), annot=True))
                plt.title("Correlation Plot")
                st.pyplot()

            
            st.header("Machine Learning Model")
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            models = []
            models.append(('LR', Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])))
            models.append(('DT', Pipeline([('scaler', StandardScaler()), ('dt', DecisionTreeClassifier())])))
            models.append(('KNN', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])))

            models_name = []
            models_mean = []
            models_std = []
            all_models = []
            scoring = 'accuracy'
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
                cv_result = model_selection.cross_val_score(model, X_test, y_test, cv=kfold, scoring=scoring)
                models_name.append(name)
                models_mean.append(cv_result.mean())
                models_std.append(cv_result.std())
                accuracy_result = {"model name":name, "model accuracy mean": cv_result.mean(), "model accuracy std": cv_result.std()}
                all_models.append(accuracy_result)

            if st.checkbox("Table"):
                st.dataframe(pd.DataFrame(zip(models_name, models_mean, models_std), columns=["Algoritma", "Mean", "Std"]))
                 
            if st.checkbox("JSON"):
                st.json(all_models)

            clf = Pipeline([('sclaer', StandardScaler()), ('dt', DecisionTreeClassifier())])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Confusion Matrix")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plot_confusion_matrix(clf, X_test, y_test)
            st.pyplot()

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

            st.subheader("Check Model")
            sepal_length = st.slider('Sepal Length', 4.3, 7.9, 5.4)
            sepal_width = st.slider('Sepal Width', 2.0, 4.4, 3.4)
            petal_length = st.slider('Petal Length', 1.0, 6.9, 1.3)
            petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

            dat = { 'sepal_length':sepal_length,
                    'sepal_width':sepal_width,
                    'petal_length':petal_length,
                    'petal_width':petal_width
            }
            features = pd.DataFrame(dat, index={0})
            st.write(features)

            prediction = clf.predict(features)
            prediction_proba = clf.predict_proba(features)

            st.subheader('Result')
            st.write(prediction)

            st.subheader('Prediction Probability')
            st.write(prediction_proba)
            



    def main(self) -> None:
        data = st.file_uploader("Upload file", type=["csv", "text"])
        self.eda(data)
    


if __name__ == '__main__':
    app = Web()
    app.main()