# Artificial Neural Network

### Importing the libraries


```python
import numpy as np
import pandas  as pd
import tensorflow as tf
```


```python
tf.__version__
```




    '2.15.0'



## Part 1 - Data Preprocessing

### Importing the dataset


```python
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
```


```python
print(X)
```

    [[619 'France' 'Female' ... 1 1 101348.88]
     [608 'Spain' 'Female' ... 0 1 112542.58]
     [502 'France' 'Female' ... 1 0 113931.57]
     ...
     [709 'France' 'Female' ... 0 1 42085.58]
     [772 'Germany' 'Male' ... 1 0 92888.52]
     [792 'France' 'Female' ... 1 0 38190.78]]



```python
print(y)
```

    [1 0 1 ... 1 1 0]


### Encoding categorical data

Label Encoding the "Gender" column


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```


```python
print(X)
```

    [[619 'France' 0 ... 1 1 101348.88]
     [608 'Spain' 0 ... 0 1 112542.58]
     [502 'France' 0 ... 1 0 113931.57]
     ...
     [709 'France' 0 ... 0 1 42085.58]
     [772 'Germany' 1 ... 1 0 92888.52]
     [792 'France' 0 ... 1 0 38190.78]]


One Hot Encoding the "Geography" column


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
```


```python
print(X)
```

    [[1.0 0.0 0.0 ... 1 1 101348.88]
     [0.0 0.0 1.0 ... 0 1 112542.58]
     [1.0 0.0 0.0 ... 1 0 113931.57]
     ...
     [1.0 0.0 0.0 ... 0 1 42085.58]
     [0.0 1.0 0.0 ... 1 0 92888.52]
     [1.0 0.0 0.0 ... 1 0 38190.78]]


### Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Part 2 - Building the ANN

### Initializing the ANN


```python
ann = tf.keras.models.Sequential()
```

### Adding the input layer and the first hidden layer


```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

### Adding the second hidden layer


```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

### Adding the output layer


```python
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## Part 3 - Training the ANN

### Compiling the ANN


```python
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Training the ANN on the Training set


```python
ann.fit(X_train, y_train, batch_size=32, epochs=80)
```

    Epoch 1/80
    250/250 [==============================] - 2s 2ms/step - loss: 0.5566 - accuracy: 0.7425
    Epoch 2/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.4678 - accuracy: 0.7960
    Epoch 3/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.4474 - accuracy: 0.7960
    Epoch 4/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.4354 - accuracy: 0.7960
    Epoch 5/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.4253 - accuracy: 0.7964
    Epoch 6/80
    250/250 [==============================] - 0s 2ms/step - loss: 0.4146 - accuracy: 0.8098
    Epoch 7/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.4043 - accuracy: 0.8181
    Epoch 8/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3927 - accuracy: 0.8249
    Epoch 9/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3812 - accuracy: 0.8319
    Epoch 10/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3729 - accuracy: 0.8380
    Epoch 11/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3673 - accuracy: 0.8435
    Epoch 12/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3633 - accuracy: 0.8490
    Epoch 13/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3607 - accuracy: 0.8531
    Epoch 14/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3585 - accuracy: 0.8545
    Epoch 15/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3571 - accuracy: 0.8555
    Epoch 16/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3561 - accuracy: 0.8579
    Epoch 17/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3544 - accuracy: 0.8580
    Epoch 18/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3537 - accuracy: 0.8585
    Epoch 19/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3523 - accuracy: 0.8593
    Epoch 20/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3510 - accuracy: 0.8604
    Epoch 21/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3505 - accuracy: 0.8612
    Epoch 22/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3497 - accuracy: 0.8605
    Epoch 23/80
    250/250 [==============================] - 0s 2ms/step - loss: 0.3488 - accuracy: 0.8615
    Epoch 24/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3483 - accuracy: 0.8614
    Epoch 25/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3478 - accuracy: 0.8627
    Epoch 26/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3470 - accuracy: 0.8614
    Epoch 27/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3464 - accuracy: 0.8625
    Epoch 28/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3462 - accuracy: 0.8629
    Epoch 29/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3456 - accuracy: 0.8608
    Epoch 30/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3449 - accuracy: 0.8610
    Epoch 31/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3447 - accuracy: 0.8612
    Epoch 32/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3444 - accuracy: 0.8634
    Epoch 33/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3437 - accuracy: 0.8624
    Epoch 34/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3437 - accuracy: 0.8635
    Epoch 35/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3432 - accuracy: 0.8631
    Epoch 36/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3427 - accuracy: 0.8631
    Epoch 37/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3425 - accuracy: 0.8625
    Epoch 38/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3426 - accuracy: 0.8619
    Epoch 39/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3419 - accuracy: 0.8618
    Epoch 40/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3420 - accuracy: 0.8629
    Epoch 41/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3412 - accuracy: 0.8625
    Epoch 42/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3415 - accuracy: 0.8630
    Epoch 43/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3412 - accuracy: 0.8640
    Epoch 44/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3405 - accuracy: 0.8631
    Epoch 45/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3414 - accuracy: 0.8627
    Epoch 46/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3405 - accuracy: 0.8615
    Epoch 47/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3407 - accuracy: 0.8643
    Epoch 48/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3398 - accuracy: 0.8616
    Epoch 49/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3400 - accuracy: 0.8622
    Epoch 50/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3393 - accuracy: 0.8616
    Epoch 51/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3399 - accuracy: 0.8615
    Epoch 52/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3391 - accuracy: 0.8631
    Epoch 53/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3394 - accuracy: 0.8629
    Epoch 54/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3391 - accuracy: 0.8629
    Epoch 55/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3385 - accuracy: 0.8620
    Epoch 56/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3386 - accuracy: 0.8635
    Epoch 57/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3384 - accuracy: 0.8646
    Epoch 58/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3382 - accuracy: 0.8622
    Epoch 59/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3380 - accuracy: 0.8633
    Epoch 60/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3380 - accuracy: 0.8635
    Epoch 61/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3378 - accuracy: 0.8636
    Epoch 62/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3378 - accuracy: 0.8625
    Epoch 63/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3384 - accuracy: 0.8622
    Epoch 64/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3378 - accuracy: 0.8650
    Epoch 65/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3374 - accuracy: 0.8636
    Epoch 66/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3370 - accuracy: 0.8630
    Epoch 67/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3373 - accuracy: 0.8631
    Epoch 68/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3373 - accuracy: 0.8639
    Epoch 69/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3365 - accuracy: 0.8639
    Epoch 70/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3372 - accuracy: 0.8622
    Epoch 71/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3369 - accuracy: 0.8633
    Epoch 72/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3363 - accuracy: 0.8614
    Epoch 73/80
    250/250 [==============================] - 1s 4ms/step - loss: 0.3366 - accuracy: 0.8626
    Epoch 74/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3368 - accuracy: 0.8641
    Epoch 75/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3361 - accuracy: 0.8630
    Epoch 76/80
    250/250 [==============================] - 1s 3ms/step - loss: 0.3366 - accuracy: 0.8639
    Epoch 77/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3367 - accuracy: 0.8634
    Epoch 78/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3363 - accuracy: 0.8620
    Epoch 79/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3359 - accuracy: 0.8633
    Epoch 80/80
    250/250 [==============================] - 1s 2ms/step - loss: 0.3360 - accuracy: 0.8626





    <keras.src.callbacks.History at 0x78f14a95ee30>



## Part 4 - Making the predictions and evaluating the model

### Predicting the result of a single observation

**Homework**

Use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: \$ 60000

Number of Products: 2

Does this customer have a credit card ? Yes

Is this customer an Active Member: Yes

Estimated Salary: \$ 50000

So, should we say goodbye to that customer ?


```python
# 데이터 입력할 때는 반드시 2D 형식으로 입력
# 현재 모델이 스케일링 된 데이터로 훈련되어 있으므로 새로 입력되는 데이터도 스케일링 해줘야 함
# 모델의 output layer에는 sigmoid 함수가 사용되므로 해당 고객이 은행을 떠날 확률을 보여줌

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```

    1/1 [==============================] - 0s 123ms/step
    [[False]]


**Solution**

Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

### Predicting the Test set results


```python
y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
```

    63/63 [==============================] - 0s 2ms/step
    [[0 0]
     [0 1]
     [0 0]
     ...
     [0 0]
     [0 0]
     [0 0]]


### Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

    [[1507   88]
     [ 189  216]]





    0.8615




