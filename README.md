# Bank Note Authentication Using Neural Networks and Random Forest

## Overview
This repository contains an implementation of **Bank Note Authentication** using **Neural Networks** and **Random Forest**. The goal is to classify whether a banknote is genuine or forged based on extracted features.

## Dataset
The dataset used for this project is the **Bank Note Authentication Dataset** from the UCI Machine Learning Repository. It contains the following attributes:
- **Variance of Wavelet Transformed Image**
- **Skewness of Wavelet Transformed Image**
- **Curtosis of Wavelet Transformed Image**
- **Entropy of Image**
- **Class (0: Forged, 1: Genuine)**

## Models Used
- **Neural Networks (ANN)**: Implemented using TensorFlow/Keras
- **Random Forest**: Implemented using Scikit-Learn

## Prerequisites
Make sure you have the following installed before running the code:
- Python (>= 3.7)
- TensorFlow/Keras
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn

### Installation
Install the required dependencies using:
```sh
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

## Getting Started
Clone this repository and navigate to the project folder:
```sh
git clone https://github.com/yourusername/Bank-Note-Authentication-Using-Neural-Networks-and-Random-Forest.git
cd Bank-Note-Authentication-Using-Neural-Networks-and-Random-Forest
```

### Running the Project
1. **Data Preprocessing and EDA**
   ```sh
   python data_preprocessing.py
   ```
2. **Train and Evaluate Neural Network**
   ```sh
   python train_nn.py
   ```
3. **Train and Evaluate Random Forest Model**
   ```sh
   python train_rf.py
   ```

## Example Code
### Loading the Dataset
```python
import pandas as pd

data = pd.read_csv("data/banknote_authentication.csv")
print(data.head())
```

### Training a Neural Network
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
```

## Project Structure
```
.
├── data/                     # Dataset
├── notebooks/                # Jupyter Notebooks
├── models/                   # Saved models
├── scripts/                  # Training and preprocessing scripts
├── README.md                 # Documentation
└── requirements.txt          # Dependencies
```

## Results
- Achieved **high accuracy** using both Neural Networks and Random Forest.
- Visualized decision boundaries and feature importance.

## Contributing
1. Fork the repository
2. Create a feature branch (`feature-name`)
3. Commit changes (`git commit -m "Added new model improvements"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions, feel free to open an issue or reach out on [LinkedIn](https://www.linkedin.com/in/your-profile).

