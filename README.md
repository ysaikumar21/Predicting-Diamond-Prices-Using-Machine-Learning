# 💎 Predicting Diamond Prices Using Machine Learning  
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen)](https://predicting-diamond-prices-using-machine-learning.streamlit.app/)

## 📌 Project Overview  
This project **predicts the price of diamonds** using **Machine Learning** models.  
The app is built using **Streamlit** and utilizes **scikit-learn** for price estimation based on diamond attributes.

🔹 **Key Features:**  
✅ Predicts **diamond price** based on weight, cut, and dimensions  
✅ Uses **trained ML models** for accurate valuation  
✅ **Live Web App** powered by **Streamlit**  
✅ **Automatically downloads ML model from Google Drive**  

---

## 🛠️ Installation & Setup  
### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/ysaikumar21/Predicting-Diamond-Prices-Using-Machine-Learning.git
cd Predicting-Diamond-Prices-Using-Machine-Learning
2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Streamlit App Locally
sh
Copy
Edit
streamlit run app.py
📂 Project Structure
bash
Copy
Edit
📂 Predicting-Diamond-Prices
│── 📄 app.py              # Streamlit app with model downloading & prediction
│── 📄 requirements.txt     # Required Python packages
│── 📂 models/              # (Model downloaded from Google Drive)
│── 📂 data/                # (Optional: Sample datasets)
🔗 Live Demo
🌍 Try the Live App Here:
🔗 Predicting Diamond Prices Using Machine Learning

📌 How the Model Works
Input Features: Carat, Cut Quality, and Size
Model Used: Pre-trained scikit-learn regression model
Google Drive Integration: If the model is missing, the app automatically downloads it from Google Drive
Example Code for Model Download:

python
Copy
Edit
import gdown
import pickle
import os

# Function to download model
def download_model():
    file_id = "1QgY6hpRoDJcuF8waxx8Ril2nhw4CjkJV"
    output = "diamond_price_model.pkl"

    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Download and load model
download_model()
with open("diamond_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
🛠️ Built With
🔹 Python
🔹 Streamlit
🔹 scikit-learn
🔹 NumPy & Pandas
🔹 gdown (for model downloads)

📧 Contact
📩 Saikumar – LinkedIn

🚀 Star this repo ⭐ and try the live demo!

yaml
Copy
Edit

---

### **✅ Next Steps**
1️⃣ **Save this file as `README.md` in your project folder**  
2️⃣ **Push it to GitHub**:  
   ```sh
   git add README.md
   git commit -m "Added README file with project details"
   git push origin main
