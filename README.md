# 🍎 Fruit Detector (Gradio App)

A simple **binary image classifier** built with **PyTorch** and **Gradio** to identify whether an uploaded image is a *fruit* or *non-fruit*.  
The model is trained using the **Fruit and Vegetable Image Recognition Dataset** from Kaggle.

---

## 🚀 Features
- Detects whether an image is **Fruit (0)** or **Non-Fruit (1)**
- Built with **PyTorch CNN**
- **Gradio UI** for easy interaction
- Lightweight and fast to deploy

---

## 🧠 Model Details
- **Architecture:** 3 Convolutional layers + 2 Fully Connected layers  
- **Input Size:** 64x64 RGB  
- **Classes:** 2 (Fruit / Non-Fruit)  
- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam (lr=0.001)

---

## 📁 Dataset
Dataset used: [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

Directory structure:
train/
validation/
test/

yaml
Copy code
Each folder contains multiple categories of fruits and vegetables.

---

## ⚙️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/kowshik-thatinati/fruit-detector-gradio.git
   cd fruit-detector-gradio
Install dependencies

bash
Copy code
pip install torch torchvision gradio
Run the Gradio app

bash
Copy code
python app.py
Open the Gradio link displayed in the terminal to test your model.

📂 File Structure
bash
Copy code
fruit-detector-gradio/
│
├── app.py                # Gradio app file
├── fruit_binary.pth      # Trained CNN model
├── README.md             # Documentation
└── requirements.txt      # Dependencies (optional)
🧩 Example Output
Input	Prediction
🍎 Apple	Fruit
🥕 Carrot	Non-Fruit

💡 Future Enhancements
Add more fine-grained classes (e.g., apple, banana, carrot)

Real-time webcam detection

Deploy to Hugging Face Spaces or Streamlit Cloud

🧑‍💻 Author
Kowshik Thatinati
Engineering Grad | AI & System Design Enthusiast
GitHub Profile

🪪 License
This project is open-source under the MIT License.
