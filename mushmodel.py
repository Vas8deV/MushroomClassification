import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

imagesize=[224,224]
trainpath=r"F:\AI Mushrooms\Dataset\Dataset\train"
testpath=r"F:\AI Mushrooms\Dataset\Dataset\test"

#testpath = os.path.abspath(testpath)

from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint

train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory(trainpath,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
test_set=test_datagen.flow_from_directory(testpath,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
xception=Xception(input_shape=imagesize + [3],
                  weights='imagenet',
                  include_top=False)
for layer in xception.layers:
    layer.trainable=False
x=Flatten()(xception.output)
prediction=Dense(3,activation='softmax')(x)
model=Model(inputs=xception.input,outputs=prediction)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
#Training model

cp_checkpoint = ModelCheckpoint("weights", 
                                save_weights_only=True,
                                verbose=1)

r=model.fit(train_set,
                      validation_data=test_set,
                      epochs=25,
                      steps_per_epoch=len(train_set)//5,
                      validation_steps=len(test_set)//5,
                      callbacks=[cp_checkpoint]
                      )
model.save('mushroom.h5')




