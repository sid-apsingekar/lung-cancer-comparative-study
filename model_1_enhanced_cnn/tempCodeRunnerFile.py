from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

print("\nClasses Found")

print(train_generator.class_indices)

print(f"\nTraining Images : {train_generator.samples}")
print(f"Validation Images : {valid_generator.samples}")
print(f"Testing Images : {test_generator.samples}")