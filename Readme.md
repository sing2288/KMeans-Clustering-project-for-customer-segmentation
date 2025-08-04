# Mall Customer Segementation App
Using K-Means clustering and Streamlit, this machine learning application divides shoppers into discrete groups.  With the help of this software, companies may gain a deeper understanding of their customers and make data-driven choices for tailored marketing and customer interaction plans.

# Features
1. Applies the K-Means algorithm using user-specified or optimal K.
2. Clusters customers into groups based on behavior or income/spending score.
3. Uses K-Means to dynamically cluster data.
4. Graphical representations to learn more about different client groupings.

# Project Structure 
Unsupervised_Clustering_Solution/
│
├── .devcontainer/
│   └── devcontainer.json              # Development container configuration
│
├── data/
│   └── Mall_Customers.csv            # Raw customer dataset
│
├── modules/
│   ├── __init__.py                   # Makes 'modules' a Python package
│   ├── data_loader.py                # Load and return the dataset
│   ├── logger.py                     # Logging utility for tracking pipeline steps
│   ├── model.py                      # Model definition and saving logic
│   ├── model.pkl                     # Trained model file
│   ├── predictor.py                  # Prediction logic using trained model
│   ├── train_model.py                # Training logic for K-Means clustering
│
├── app.py                            # Main script or Streamlit app entry point
├── requirements.txt                  # List of dependencies
├── README.md                         # Project overview and usage instructions
├── .gitignore.txt                    # Git ignore rules for unnecessary files
├── .gitattributes.txt                # Git attributes for handling text and binary files

# How to run the project
1. Clone the repository 
git clone https://github.com/sing2288/Unsupervised_Clustering_Solution.git
cd Unsupervised_Clustering_Solution

2. Install dependencies
pip install -r requirements.txt

3. Run the app 
Streamlit run app.py 

# Dependencies 
streamlit
pandas
seaborn
matplotlib
scikit-learn
logging 

# License 
The Project is made for eductaional purposes only.