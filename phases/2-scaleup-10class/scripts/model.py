"""
Step 3 — THE MODEL
==================

This builds the paper's "VGG-style" 1D CNN (convolutional neural network),
scaled to our small 3-class problem.

The idea of the design (from the VGG family of networks):
  * Use many SMALL convolution filters (size 3) stacked in a row.
  * After each one, HALVE the length with max-pooling.
  * Do this until the signal is short, then hand it to a few dense layers
    that make the final decision.

A "convolution" here is a small filter that slides along the 1024-step signal
looking for a local pattern (like a little shape in the waveform). Stacking
them lets later layers combine simple shapes into more complex ones.

Layer-by-layer (input is 1024 time-steps x 2 channels = I/Q):

    Input                       1024 x 2
    7 x [ Conv1D(64, size 3, ReLU) + BatchNorm + MaxPool/2 ]
                                length: 1024 -> 512 -> 256 -> ... -> 8
    Flatten                     turn the grid into one long list of numbers
    Dense(128, SELU) + AlphaDropout
    Dense(128, SELU) + AlphaDropout
    Dense(3, Softmax)           -> 3 probabilities, one per modulation

Reminders on the pieces (all covered earlier):
  * ReLU        = keep positives, zero negatives (the standard activation).
  * BatchNorm   = tidy the numbers between layers so training is fast/stable.
  * SELU        = self-tidying activation used in the dense layers; it needs
                  the 'lecun_normal' starting weights and AlphaDropout to work.
  * Softmax     = squashes the final 3 numbers into probabilities that add to 1.
  * Dropout     = randomly ignores some numbers during training so the model
                  doesn't just memorize; AlphaDropout is the SELU-friendly version.

Run `.venv/bin/python scripts/model.py` on its own to just print the summary.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

import keras
from keras import layers


def build_vgg_cnn(input_shape=(1024, 2), n_classes=3, n_conv_blocks=7, filters=64):
    """Create and return the (uncompiled) VGG-style 1D CNN."""
    model = keras.Sequential(name="vgg_cnn_3mod")
    model.add(keras.Input(shape=input_shape))

    # --- The convolutional feature-extractor: repeat the same block ---
    for i in range(n_conv_blocks):
        model.add(layers.Conv1D(filters, kernel_size=3, padding="same",
                                activation="relu", name=f"conv{i+1}"))
        model.add(layers.BatchNormalization(name=f"bn{i+1}"))
        model.add(layers.MaxPooling1D(pool_size=2, name=f"pool{i+1}"))
        # length halves each block: 1024 -> 512 -> ... -> 8

    # --- Flatten the (length x filters) grid into one long vector ---
    model.add(layers.Flatten(name="flatten"))

    # --- The classifier: two SELU dense layers, then the decision layer ---
    for i in range(2):
        model.add(layers.Dense(128, activation="selu",
                               kernel_initializer="lecun_normal",
                               name=f"dense{i+1}"))
        model.add(layers.AlphaDropout(0.1, name=f"drop{i+1}"))

    # Final layer: one output per class, softmax => probabilities summing to 1
    model.add(layers.Dense(n_classes, activation="softmax", name="output"))
    return model


if __name__ == "__main__":
    # Running this file directly just shows the architecture and parameter count.
    m = build_vgg_cnn()
    m.summary()
