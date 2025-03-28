# ECG Classification with PyTorch

## 🚀 Project Overview
This project demonstrates ECG signal classification using a simple neural network built with PyTorch. It uses the ECG200 dataset from the UCR repository, training a neural network model to classify ECG signals.

## 📂 Project Structure
```
ECG+Pytorch
├── UCR_Dataset
│   ├── ECG200_TEST.arff
│   ├── ECG200_TEST.ts
│   ├── ECG200_TEST.txt
│   ├── ECG200_TRAIN.arff
│   ├── ECG200_TRAIN.ts
│   ├── ECG200_TRAIN.txt
│   └── ECG200.txt
├── 1st.py
└── README.md
```

## 📝 Dataset
- **ECG200_TRAIN.txt** and **ECG200_TEST.txt** contain ECG signals.
- First column: Labels (1 for normal, -1 for abnormal)
- Remaining columns: ECG signal values

## 🛠️ Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- PyTorch

Install dependencies using:
```
pip install numpy pandas matplotlib scikit-learn torch torchvision
```

## 💻 Running the Code
1. Clone the repository and navigate to the project folder.
2. Ensure dataset files are in the correct location.
3. Run the script:
```
python 1st.py
```

## ⚙️ Model Overview
A simple neural network with:
- Input Layer: 96 neurons (ECG features)
- Hidden Layer: 64 neurons with ReLU activation
- Output Layer: 2 neurons (Binary classification)

## 📊 Results
The training loop runs for 20 epochs, and the model evaluates on the test set. Accuracy is calculated and printed.

## 🖼️ Sample ECG Plots
The code visualizes sample ECG signals from the training set.

## 🟢 Future Enhancements
- Experiment with deeper neural networks
- Implement other classification algorithms
- Tune hyperparameters

## 🤝 Contributing
Feel free to open issues or submit pull requests.

## 📜 License
This project is open-source and available under the MIT License.

---
Happy coding! 😊

