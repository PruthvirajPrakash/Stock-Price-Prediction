**Dataset Description**
 The dataset may have been collected from medical records or a health survey.

Fields
The dataset includes various symptom-related fields such as:

itching
skin_rash
nodal_skin_eruptions
continuous_sneezing
shivering
chills
joint_pain
stomach_pain
acidity
ulcers_on_tongue
It also includes a field named prognosis, which is used as the target variable for predictions. This field likely contains the diagnosis or the outcome that corresponds to the combination of symptoms.

Purpose and Utility
The primary utility of this dataset appears to be in developing predictive models to diagnose medical conditions based on symptom data. This could be incredibly useful in healthcare settings to assist in preliminary diagnosis or in triage systems to direct patients to the appropriate care based on their symptoms.

**EDA**

Frequency Vs Symptoms


![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/3c2fd917-4e5e-4a9b-8191-df142933fb9a)

**Corelation plot**


![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/07f86473-f9ce-43cd-a0db-1c0b4fd2ecd5)



**Features and Target Variables**

X Variables (Features)
Itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain.
Y Variable (Target)
y is the target variable, which in this case is the prognosis.


**Feature Engineering:**
Feature engineering to selecting relevant features and dropping others

**Model Selecting**
Two models were selected: a Decision Tree Classifier and a Random Forest Classifier. The Decision Tree was chosen for its simplicity and interpretability, useful for understanding the impact of individual features on the diagnosis. The Random Forest was chosen for its robustness and better generalization, as it reduces overfitting by averaging multiple decision trees. These models are often effective for classification tasks with structured data like symptoms and diagnosis.

**Model Fitting & Train/Test Splitting**
Split of training and testing data using train_test_split with a typical random allocation. For the Decision Tree model, a test size of 20% was chosen, and for the Random Forest, it was 40%, likely to validate model robustness.
**Data Leakage Risks**: 
There's no specific handling of data leakage evident,
**Model Selection & Hyperparameter Tuning:**
Decision Tree and Random Forest classifiers were selected, presumably for their efficacy in handling categorical and non-linear relationships in data. 

**Validation / metrics**

DecisionTreeClassifier


![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/4db5d7c1-de5d-4210-a789-ed50b75a35af)


Randomforest

RandomForestClassifier


![image](https://github.com/PruthvirajPrakash/Stock-Price-Prediction/assets/152721488/b32566cb-7ea6-4bd9-8c59-f5801e415b7c)

**Production**

**Advice for Deployment:**
The model should be deployed into a clinical decision support system with an intuitive interface that allows healthcare professionals to input symptoms and receive diagnostic predictions. It’s essential to maintain, update, and monitor the model's performance continuously to ensure reliability.

**Precautions about its Use**:
It's crucial to provide clear disclaimers about the model's limitations and ensure it is used as an aid, not a replacement for professional medical diagnosis.
The potential for misdiagnoses should be mitigated by involving healthcare professionals in the validation process of diagnoses offered by the model.

**Going Further**

Improvements for the Model:
Enhancing the model could involve collecting more varied data from a broader patient demographic to improve its generalizability. Including additional features such as patient history, lifestyle factors, and genetic information could provide a more holistic view and improve diagnostic accuracy.

**Technological Enhancements:**
Applying more robust machine learning techniques like ensemble methods.
Improve model by using advanced scaling methods.













 
