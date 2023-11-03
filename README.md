# midterm_project_ml_zoomcamp
![photo_for_the_notebook](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/e0997525-0c58-4854-adf5-de30abf4c86c)

Link for the dataset: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data

My Kaggle Profile: https://www.kaggle.com/lokmanefe

My GitHub Profile: https://github.com/lokicik

My LinkedIn Profile: https://www.linkedin.com/in/lokmanefe/

Feel free to reach out! ✌

### **Project Presentation**
I would like to introduce my most comprehensive project to date! I have completed the project "Drinking or Smoking Classification" which I developed as the Midterm Project during the 7th week of the Machine Learning Zoomcamp organized by DataTalksClub and presented by Alexey Grigorev. I had the opportunity to apply all the techniques I learned, and I gave my all to this project. I look forward to your feedback and hope for positive reviews!
### **Problem Description**
Besides regular users of smoke and alcohol, passive smokers and non-regular drinkers are getting their body hurt aswell. This project can be used to determine if a passive smoker or a non-regular drinker doing how much damage to their body and know about if they are smoking or drinking as much as a regular users, so they can lower their usage of drinking or smoking. This project also can be used to determine if a child is smoking or drinking, his/her body is affected by the smokers or drinkers around him/her or not.
### **Problem Solution**
I've used the dataset to develop 2 different models, one for Alcohol Drinking Prediction, and one for Smoking Prediction. The reason I did this is so that the user can input a single record and get the results from both models, drinking and smoking probability. The target variable for the drinking model is "DRK_YN",  for the smoking model it's "SMK_stat_type_cd". I've  extracted the target variables from the dataset and trained models separately. The Smoking Model is a multiclass -can be binary if predictions for 0 and 1 is summarized in prediction 0, which would mean not drinker, and 1 stays 1 which already stands for drinker- classification model which predicts 0/1/2 (never smoked/smoked but quit/smoker). The Drinking Model is a binary classification model which predicts 0/1(drinker/not drinker).

### **Dataset**
This dataset is collected from National Health Insurance Service in Korea. All personal information and sensitive data were excluded.
The purpose of this dataset is to:

*   Analysis of body signal
*   Classification of smoker or drinker

  
**Details of dataset:**

*   Sex	male, female
*   age	round up to 5 years
*   height	round up to 5 cm[cm]
*   weight	[kg]
*   sight_left	eyesight(left)
*   sight_right	eyesight(right)
*   hear_left	hearing left, 1(normal), 2(abnormal)
*   hear_right	hearing right, 1(normal), 2(abnormal)
*   SBP	Systolic blood pressure[mmHg]
*   DBP	Diastolic blood pressure[mmHg]
*   BLDS	BLDS or FSG(fasting blood glucose)[mg/dL]
*   tot_chole	total cholesterol[mg/dL]
*   HDL_chole	HDL cholesterol[mg/dL]
*   LDL_chole	LDL cholesterol[mg/dL]
*   triglyceride	triglyceride[mg/dL]
*   hemoglobin	hemoglobin[g/dL]
*   urine_protein	protein in urine, 1(-), 2(+/-), 3(+1), 4(+2), 5(+3), 6(+4)
serum_creatinine	serum(blood) creatinine[mg/dL]
*   SGOT_AST	SGOT(Glutamate-oxaloacetate transaminase) AST(Aspartate transaminase)[IU/L]
*   SGOT_ALT	ALT(Alanine transaminase)[IU/L]
*   gamma_GTP	y-glutamyl transpeptidase[IU/L]

*   SMK_stat_type_cd	Smoking state, 1(never), 2(used to smoke but quit), 3(still smoke)
*   DRK_YN	Drinker or Not

### **Midterm Project Requirements (Evaluation Criteria)**

* Problem description
* EDA
* Model training
* Exporting notebook to script
* Model deployment
* Reproducibility
* Dependency and environment management
* Containerization
* Cloud deployment


### **Dependency and environment management guide**

``pip install pipenv``

``pip shell``

``pip install -r requirements.txt``


### **Containerization**
1-)``docker tag image_name YOUR_DOCKERHUB_NAME/image_name``
![docker1](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/d2303b74-b71d-45f9-9d2d-e03ec7b1cc1c)


2-)``docker push YOUR_DOCKERHUB_NAME/image_name``
![docker2](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/a0bafe9c-24bd-484e-9111-36926d1007ed)


Useful link encountered with error.
https://www.biostars.org/p/9531985/#:~:text=As%20the%20error%20says%20requested,docker%20hub%20credentials%20are%20incorrect.

### **Deployment guide**
#### **To run it locally:**

* Run ``python predict.py`` on a terminal

* Open a new terminal and run python ``predict_docker.py``

#### **To run it on docker:**

* Open a terminal

* ``docker build -t midterm_project .``

* ``docker run -it --rm -p 7860:7860 midterm_project``

* Open a new terminal and run ``python predict_docker.py``

#### **To run it on cloud:**

* Push your image to docker hub like in the Containerization section

* Open https://render.com/ and create a web service

* Select "Deploy an existing image from a registry"
![render0](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/e07f0a30-4aad-450e-8a77-7c82838c746e)

* Enter the image url for your "YOUR_DOCKERHUB_NAME/image_name"
![render1](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/31f2479b-a562-4142-a2c0-89b0838ae5b0)

* Finalize the setup and run your web service

![render2](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/b15fb9f9-ed32-4eb9-9078-cb2418db4eb9)

* Open up a new terminal and run ``python predict_cloud.py``

* Here is the final result
  
![result](https://github.com/lokicik/midterm_project_ml_zoomcamp/assets/65876412/c549dcbc-7c98-4bcf-9535-82ad628f8b9b)









