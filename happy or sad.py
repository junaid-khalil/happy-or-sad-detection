from sklearn._loss import loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Data preparation

training = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

training_dataset = training.flow_from_directory('comp/New folder/training/', target_size=(200, 200), batch_size=4,
                                                class_mode='binary')
validation_dataset = training.flow_from_directory('comp/New folder/validation/', target_size=(200, 200), batch_size=4,
                                                  class_mode='binary')

# Model definition

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    #tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                    #tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    ##tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    ##tf.keras.layers.MaxPool2D(2, 2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(512, activation='relu',),
                                    tf.keras.layers.Dropout(0.5),

                                    #
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                    ])
#model.summary()
# Model compilation
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# Model training
model_fit = model.fit(training_dataset,
                      steps_per_epoch=5,
                      epochs=30,
                      validation_data=validation_dataset)

#model.save('happy.h5')



#plotting the evaluation

h=model_fit
#plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()
# plot the accuracy value
plt.plot(h.history['accuracy'], label='train accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()


# Perform predictions
dir_path = 'comp/New folder/testing'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print('happy')
    else:
        print('sad')