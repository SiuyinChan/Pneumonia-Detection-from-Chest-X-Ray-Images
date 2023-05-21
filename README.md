# Pneumonia Detection from Chest X-Ray Images using Machine Learning Models
This repository contains the code for three machine learning models developed for the detection of pneumonia based on x-ray images. The models included are Convolutional Neural Network (CNN), Random Forest, and Gradient Boosting.

## Dataset
You can download the dataset from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/), and place it in the `data` folder.

The number of `Normal` and `Pneumonia` images in the 3 datasets:

| Dataset Type | Normal | Pneumonia |
|--------------|--------|-----------|
| Training     | 1341   | 3875      |
| Validation   | 8      | 8         |
| Test         | 234    | 390       |

## Dependencies
Install the dependencies:
```
pip install -r requirements.txt
```

## Load models
To load saved models from the root of the directory:

• CNN model:
```
from tensorflow import keras
cnn_model = keras.models.load_model('./models/cnn')
```

• Random Forest model:
```
import joblib
random_forest_model = joblib.load('./models/random_forest.joblib')
```

• Gradient Boosting model:
```
import joblib
gradient_boosting_model = joblib.load('./models/gradient_boosting.joblib')
```

## Results
The performance of each model on the test dataset is as follows:

- ### CNN Model

![CNN Model](./images/cnn.png?raw=true)

- ### Random Forest Model:

![Random Forest Model](./images/random_forest.png?raw=true)

- ### Gradient Boosting Model:

![Gradient Boosting Model](./images/gradient_boosting.png?raw=true)

## Conclusion
Based on the results obtained, the CNN model demonstrated the highest performance in pneumonia detection from x-ray images. The Random Forest and Gradient Boosting models also showed respectable performance, albeit slightly lower than the CNN model. It is recommended to utilize the CNN model for pneumonia detection tasks.

Please refer to the individual model scripts for more details on the model architectures, training, and evaluation procedures.

## Future Work
To further improve the models' performance and applicability, consider the following future work:

* Collecting and incorporating a larger and more diverse dataset.
* Exploring transfer learning techniques for the CNN model using pre-trained models.
* Conducting hyperparameter tuning to optimize the models' performance.
* Exploring interpretability techniques to gain insights into the models' decision-making process.