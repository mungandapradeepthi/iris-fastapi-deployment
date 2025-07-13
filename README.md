Project Summary: Deploying an Iris Classifier Using FastAPI

Project Overview:
This project focuses on deploying a machine learning model as a REST API using FastAPI, a modern, high-performance Python web framework for building APIs. The model classifies Iris flower species based on petal and sepal measurements from the well-known Iris dataset. The solution demonstrates how to train a model using Scikit-learn, save it using joblib, and build a FastAPI server to expose the prediction capability via a /predict endpoint.

Dataset Description:
•	Source: sklearn.datasets.load_iris()
•	Features:
o	Sepal length
o	Sepal width
o	Petal length
o	Petal width
•	Target classes:
o	Setosa (0)
o	Versicolor (1)
o	Virginica (2)
•	Total Samples: 150
The dataset is split into training and testing subsets to evaluate model performance.

 Model Development
A Logistic Regression model is used for multi-class classification. It was selected for its interpretability and effectiveness for linearly separable datasets like Iris.
Key Steps:
1.	Load the Iris dataset.
2.	Split the data using train_test_split().
3.	Train the model with LogisticRegression().
4.	Save the trained model using joblib.dump() for later inference.
 The model achieved high accuracy on the test set and was saved as iris_model.pkl.

 FastAPI Deployment
 File Structure
project/
│
├── iris_model.pkl
├── main.py
└── requirements.txt
 main.py (FastAPI App)
•	Model Loading: The saved model is loaded during app startup using joblib.load().
•	Input Validation: A Pydantic model ensures clean and typed input data.
•	Endpoint: /predict accepts POST requests with input measurements and returns the predicted species.
 Sample Request:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
 Sample Response:
{
  "prediction": "setosa"
}
The API can be tested via Swagger UI or external tools like Postman or CURL.
 Real-World Applications:
This project simulates a real-world use case where businesses or research institutions need to automatically classify Iris species. Similar deployments can be applied to:
•	Plant classification in agriculture
•	Inventory management in nurseries
•	Educational tools for botany students

 Key Learnings
•	Gained practical experience deploying machine learning models as web services.
•	Learned how to structure and build a REST API using FastAPI.
•	Applied model serialization with joblib.
•	Built a reusable API endpoint for prediction, supporting real-time inference.

 Conclusion:
This project successfully demonstrates the end-to-end pipeline from training a machine learning model to serving it through a production-ready FastAPI endpoint. The deployment is modular, scalable, and beginner-friendly, offering a foundation for further enhancement such as Dockerization, CI/CD pipelines, or cloud deployment.

