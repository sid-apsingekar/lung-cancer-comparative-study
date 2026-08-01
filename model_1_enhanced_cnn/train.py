import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from model import build_enhanced_cnn
from dataset import (
    train_generator,
    valid_generator
)

from config import *

model = build_enhanced_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_DIR / "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

model.save(CHECKPOINT_DIR / "final_model.keras")



history_df = pd.DataFrame(history.history)

history_df.to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)