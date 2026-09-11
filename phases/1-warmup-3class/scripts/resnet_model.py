"""
Step 6 — THE RESNET (the paper's best model)
============================================

This builds the paper's RESIDUAL network (Table IV + Figure 5), the model that
beat the VGG CNN in their results (99.8% vs 98.3% at high SNR on 24 classes).

WHAT'S ACTUALLY NEW HERE: the SKIP CONNECTION
---------------------------------------------
In our VGG, data flows in a straight line: each layer feeds the next.

    x -> conv -> conv -> conv -> ...

A residual unit adds a shortcut that jumps OVER a couple of layers:

    x ---------------------------.
     \                            \
      -> conv(ReLU) -> conv(linear) -> (+) -> out

So instead of  out = layers(x),  we compute  out = layers(x) + x.

Why that tiny "+ x" matters:
  * The layers only have to learn the CHANGE (the "residual") needed, not
    rebuild the whole signal from scratch. Learning "adjust this slightly" is
    much easier than learning "reproduce all of this, plus adjust it".
  * If a block isn't useful, the network can just learn to output roughly zero
    and the shortcut passes the signal through untouched. Extra depth can no
    longer HURT you, which is why very deep networks became trainable.
  * Training signal (the gradient) gets a clean path straight back through the
    "+" to the earlier layers, instead of being squeezed through every layer.

This idea is the backbone of nearly every modern deep network built since 2015,
including the transformers behind today's large language models.

BECAUSE OF THE SKIP, WE CAN'T USE Sequential
--------------------------------------------
`keras.Sequential` only expresses a straight chain. To make a value fork off and
rejoin later we use the FUNCTIONAL API: we hold layer outputs in variables and
wire them up by hand, then say `keras.Model(inputs, outputs)`.

THE LAYOUT (paper's Table IV / Figure 5)
-----------------------------------------
    Input                       1024 x 2
    6 x Residual Stack          length 1024 -> 512 -> ... -> 16, 32 filters
    Flatten
    Dense(128, SELU) + AlphaDropout
    Dense(128, SELU) + AlphaDropout
    Dense(3, Softmax)

where one Residual Stack is:

    Conv1D(32, kernel 1, linear)   <- resizes the channel count
    Residual Unit                  <- conv/conv + skip
    Residual Unit                  <- conv/conv + skip
    MaxPool /2                     <- halve the length, same as the VGG

Note it uses 32 filters where our VGG used 64. The residual structure gets more
out of each filter, so the paper uses fewer and still lands at a comparable
parameter count -- which makes this a fair head-to-head comparison.

Run `.venv/bin/python scripts/resnet_model.py` to just print the summary.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hush TF startup noise

import keras
from keras import layers


def residual_unit(x, filters, name):
    """Two convolutions, with the input added back on at the end (the skip)."""
    shortcut = x                      # remember what came in

    # First conv: ReLU, as in the paper's Figure 5.
    y = layers.Conv1D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu")(y)

    # Second conv: LINEAR (no activation) so the sum below isn't distorted.
    y = layers.Conv1D(filters, 3, padding="same", name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    # THE SKIP CONNECTION: add the original input back onto the result.
    return layers.Add(name=f"{name}_add")([shortcut, y])


def residual_stack(x, filters, name):
    """A 1x1 conv to set the channel count, two residual units, then pool."""
    # kernel_size=1 with no activation: this touches each time-step on its own,
    # purely to change how many channels there are (2 -> 32 on the first stack).
    # It's needed so the shortcut and the conv output have matching shapes to add.
    x = layers.Conv1D(filters, 1, padding="same", name=f"{name}_reshape")(x)

    x = residual_unit(x, filters, name=f"{name}_unit1")
    x = residual_unit(x, filters, name=f"{name}_unit2")

    return layers.MaxPooling1D(pool_size=2, name=f"{name}_pool")(x)


def build_resnet(input_shape=(1024, 2), n_classes=3, n_stacks=6, filters=32):
    """Create and return the (uncompiled) residual network."""
    inputs = keras.Input(shape=input_shape, name="input")

    x = inputs
    for i in range(n_stacks):
        x = residual_stack(x, filters, name=f"stack{i+1}")
        # length halves each stack: 1024 -> 512 -> ... -> 16

    x = layers.Flatten(name="flatten")(x)

    # Same classifier head as the VGG, so the only real difference between the
    # two models is the convolutional part.
    for i in range(2):
        x = layers.Dense(128, activation="selu",
                         kernel_initializer="lecun_normal",
                         name=f"dense{i+1}")(x)
        x = layers.AlphaDropout(0.1, name=f"drop{i+1}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    # The functional API: hand Keras the start and end, it works out the graph.
    return keras.Model(inputs, outputs, name="resnet_3mod")


if __name__ == "__main__":
    m = build_resnet()
    m.summary()
    print(f"\nTotal layers: {len(m.layers)}")
