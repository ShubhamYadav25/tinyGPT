
# ============================================================
# STEP 1: PREPARE OUR DATASET
# Goal: Convert text into numbers that a model can understand.
# ============================================================

# 1. Our tiny training dataset.
# The model will learn patterns from these example words.
text = """
hello
hello
help
he
"""

print("Original dataset:")
print(text)


# 2. Extract every unique character from the text.
# set() removes duplicates.
# sorted() puts the characters in a consistent order.
vocabulary = sorted(set(text))

print("Unique characters:")
print(vocabulary)

# 3. Count how many unique characters exist.
vocab_size = len(vocabulary)

print("\nVocabulary size:", vocab_size)


# 4. Assign a unique integer ID to each character.
# Example: 'a' might become 0, 'b' might become 1.
# The actual IDs depend on our sorted vocabulary.
char_to_id = {
    character: index for index, character in enumerate(vocabulary) }

# We also need the reverse mapping:
# ID -> original character.
id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

print("\nCharacter to ID:")
print(char_to_id)


# 5. ENCODING: Convert text into a list of integer IDs.
# Every character is replaced with its assigned number.
encoded_text = [
    char_to_id[character]
    for character in text
]

print("\nFirst 20 encoded IDs:")
print(encoded_text[:20])


# 6. DECODING: Convert integer IDs back into characters.
# This lets us reconstruct the original text.
decoded_text = "".join(
    id_to_char[token_id]
    for token_id in encoded_text
)

print("\nDecoded text:")
print(decoded_text)

# 7. Verify that encoding and decoding worked correctly.
assert decoded_text == text

print("Round-trip check passed!")

#============================================ Train the model to predict the next character in a sequence. ===========================================
"""
Training examples
Input (context)
h

Correct next character
e


Input (context)
he

Correct next character
l


Input (context)
hel

Correct next character
l


Input (context)
hell

Correct next character
o
"""


# ============================================================
# STEP 2: CREATE INPUT-TARGET TRAINING EXAMPLES
# Goal: Teach the model to predict the next character.
# ============================================================

# Our tiny example dataset.
text = "hello"

# How many previous characters can the model see?
# For now, we will allow the context to grow.
# Later, we will use a fixed context length.
context_length = 3

print("Training text:", text)
print("\nTraining examples:\n")

# Each position gives us a context and its next character.
for i in range(1, len(text)):

    # Take all characters before position i.
    # Example: when i = 3, context = "hel".
    context = text[:i]

    # The character at position i is the correct answer.
    # Example: when i = 3, target = "l".
    target = text[i]

    print(f"Input: {context!r:8} -> Target: {target!r}")


print("\n" + "=" * 40)
print("Now use a fixed context length:", context_length)
print("=" * 40)

# A model commonly receives a limited number of characters.
# We slide this fixed-size window across the text.
for i in range(len(text) - context_length):

    # Select three consecutive characters as the input.
    context = text[i:i + context_length]

    # The next character becomes the target.
    target = text[i + context_length]

    print(f"Input: {context!r} -> Target: {target!r}")

#=============================================== Convert training examples into numbers

# ============================================================
# STEP 3: PREPARE NUMERICAL TRAINING DATA
# Goal: Create X (inputs) and Y (correct next-character IDs).
# ============================================================

import numpy as np

# 1. Training text.
text = "hello"

# 2. Create our vocabulary.
# Sorting ensures that character IDs are reproducible.
vocabulary = sorted(set(text))

# Convert each character to an integer ID.
char_to_id = {
    char: index
    for index, char in enumerate(vocabulary)
}

# Reverse mapping: integer ID -> character.
id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

print("Vocabulary:", vocabulary)
print("Character IDs:", char_to_id)


# 3. Encode the entire text as integers.
# Example: "hello" -> [1, 0, 2, 2, 3]
encoded = np.array([
    char_to_id[char]
    for char in text
])

print("\nEncoded text:", encoded)


# 4. Choose how many characters the model can see.
context_length = 3

# X stores the input character IDs.
# Y stores the ID of the correct next character.
X = []
Y = []

# Slide a window across the encoded sequence.
# Stop when there is no character left to predict.
for i in range(len(encoded) - context_length):

    # Take three consecutive IDs as the input.
    input_ids = encoded[i:i + context_length]

    # The next ID is the correct target.
    target_id = encoded[i + context_length]

    X.append(input_ids)
    Y.append(target_id)


# 5. Convert the lists into NumPy arrays.
# Arrays make numerical operations easier later.
X = np.array(X, dtype=np.int64)
Y = np.array(Y, dtype=np.int64)

print("\nInput array X:")
print(X)

print("\nTarget array Y:")
print(Y)


# 6. Inspect the shape of our data.
# Shape tells us the dimensions of an array.
print("\nShape of X:", X.shape)
print("Shape of Y:", Y.shape)


# 7. Decode the examples to check our work.
print("\nHuman-readable training examples:")

for input_ids, target_id in zip(X, Y):

    # Convert the input IDs back into characters.
    input_text = "".join(
        id_to_char[int(token_id)]
        for token_id in input_ids
    )

    # Convert the target ID back into its character.
    target_char = id_to_char[int(target_id)]

    print(f"Input: {input_text!r} -> Target: {target_char!r}")

"""
Encoded text: [1 0 2 2 3]

Input array X:
[[1 0 2]
 [0 2 2]]

Target array Y:
[2 3]

Shape of X: (2, 3) => means we have 2 examples, with 3 input characters per example.
Shape of Y: (2,) => means we have 2 target values, one for each example.
"""

#=============================================== 
"""
tensor : A tensor is a general term for data arranged in one or more dimensions.

logits = X.W+b
\(X\) — the numerical input.
\(W\) — weights that control how inputs influence the output.
\(b\) — biases that shift the output scores.
Logits — raw scores for each possible next character.

Initially, we'll set the weights and biases to small random values. Later, training will adjust them so the model makes better predictions.
we'll convert each character ID into a one-hot vector first. This prevents the model from mistakenly treating character ID 3 as mathematically greater than character ID 1.

"""

# ============================================================
# Goal: Turn an input context into scores for the next character.
# ============================================================

import numpy as np

# Fix the random seed so that you get repeatable results.
np.random.seed(42)

# Our vocabulary and character-to-ID mapping.
vocabulary = sorted(set("hello"))
char_to_id = {ch: i for i, ch in enumerate(vocabulary)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

vocab_size = len(vocabulary)
context_length = 3

print("Vocabulary:", vocabulary)
print("Vocabulary size:", vocab_size)


# ------------------------------------------------------------
# PART 1: Prepare a sample input: "hel"
# ------------------------------------------------------------

input_text = "hel"

# Convert each character into its integer ID.
input_ids = [char_to_id[ch] for ch in input_text]

print("\nInput text:", input_text)
print("Input IDs:", input_ids)


# ------------------------------------------------------------
# PART 2: One-hot encode each character
# ------------------------------------------------------------

# A one-hot vector has one 1 and zeros everywhere else.
# Each vector has one position for every vocabulary character.
one_hot = np.eye(vocab_size, dtype=np.float32)

# Look up the one-hot vector for every input character.
encoded_vectors = one_hot[input_ids]

print("\nOne-hot input shape:", encoded_vectors.shape)
print(encoded_vectors)


# ------------------------------------------------------------
# PART 3: Flatten the three vectors into one input vector
# ------------------------------------------------------------

# Three characters, each represented by vocab_size numbers.
# Flatten combines them into one long vector.
x = encoded_vectors.flatten()

input_size = context_length * vocab_size

print("\nFlattened input:")
print(x)
print("Input vector shape:", x.shape)


# ------------------------------------------------------------
# PART 4: Create the model's trainable parameters
# ------------------------------------------------------------

# W connects every input feature to every output character.
# Rows correspond to input features.
# Columns correspond to possible next characters.
#
# Shape: (input_size, vocab_size)
W = np.random.randn(input_size, vocab_size) * 0.1

# One bias for each possible output character.
# Initially, all biases are zero.
b = np.zeros(vocab_size, dtype=np.float32)

print("\nWeight matrix shape:", W.shape)
print("Bias vector shape:", b.shape)


# ------------------------------------------------------------
# PART 5: Make a forward pass
# ------------------------------------------------------------

# Matrix multiplication combines input values with weights.
# Adding the bias produces the raw prediction scores (logits).
logits = x @ W + b

print("\nLogits:", logits)
print("Logits shape:", logits.shape)


# ------------------------------------------------------------
# PART 6: Find the character with the highest score
# ------------------------------------------------------------

# argmax returns the index of the largest score.
predicted_id = int(np.argmax(logits))

# Convert that output ID back into a character.
predicted_char = id_to_char[predicted_id]

print("\nPredicted next character:", predicted_char)
print("Correct next character for 'hel':", "l")

#===============================================
"""
Softmax: turn scores into probabilities
Next character	Logit
e	1.0
h	0.5
l	2.0
o	-0.5
The largest score belongs to l, which is the correct next character for hel. However, these scores are not probabilities: they don't add up to 1.
Softmax converts the scores into probabilities between 0 and 1, with all probabilities summing to 1.

Cross-entropy loss: measure the mistake

"""


# ============================================================
# STEP 5: SOFTMAX AND CROSS-ENTROPY LOSS
# Goal: Convert scores to probabilities and measure mistakes.
# ============================================================

import numpy as np

# The vocabulary order is important:
# ID 0 = e, ID 1 = h, ID 2 = l, ID 3 = o
vocabulary = ["e", "h", "l", "o"]

# Example scores produced by our neural network.
# These are illustrative values, not trained model outputs.
logits = np.array([1.0, 0.5, 2.0, -0.5])

# For the input "hel", the correct next character is "l".
target_id = 2

print("Logits:", logits)


# ------------------------------------------------------------
# PART 1: Calculate softmax probabilities
# ------------------------------------------------------------

def softmax(logits):
    # Subtracting the largest logit improves numerical stability.
    # This does not change the resulting probabilities.
    shifted_logits = logits - np.max(logits)

    # Exponentials make all values positive.
    exp_values = np.exp(shifted_logits)

    # Divide by the total to make the probabilities sum to 1.
    probabilities = exp_values / np.sum(exp_values)

    return probabilities


probabilities = softmax(logits)

print("\nProbabilities:")

for char, probability in zip(vocabulary, probabilities):
    print(f"{char}: {probability:.4f} ({probability * 100:.2f}%)")

print("Probability sum:", probabilities.sum())


# ------------------------------------------------------------
# PART 2: Calculate cross-entropy loss
# ------------------------------------------------------------

# Select the probability assigned to the correct character.
correct_probability = probabilities[target_id]

# A high correct probability means low loss.
# A low correct probability means high loss.
loss = -np.log(correct_probability)

print("\nCorrect character:", vocabulary[target_id])
print("Correct probability:", round(correct_probability, 4))
print("Cross-entropy loss:", round(loss, 4))


# ------------------------------------------------------------
# PART 3: Compare good and bad predictions
# ------------------------------------------------------------

# These examples are already probabilities, not logits.
for p in [0.99, 0.80, 0.50, 0.10, 0.01]:

    # Calculate the loss when the correct answer gets probability p.
    example_loss = -np.log(p)

    print(f"P(correct) = {p:.2f} -> Loss = {example_loss:.3f}")

#============================================================== Backpropagation and gradient descent
"""
When the model makes a mistake, how does it know which weights to change, and by how much?

Gradient
A gradient tells us how the loss changes when a parameter changes.
- If a weight gradient is −16. It tells us that increasing this weight will reduce the loss, at least locally.

"""

# ============================================================
# STEP 6: BACKPROPAGATION + ONE GRADIENT DESCENT UPDATE
# Goal: Update our text model so "l" becomes more likely
# after the context "hel".
# ============================================================

import numpy as np

np.random.seed(42)

# ------------------------------------------------------------
# PART 1: Prepare the input and vocabulary
# ------------------------------------------------------------

vocabulary = sorted(set("hello"))
char_to_id = {ch: i for i, ch in enumerate(vocabulary)}

vocab_size = len(vocabulary)

# We want the model to predict the next character after "hel".
input_text = "hel"
target_char = "l"

# Convert the input characters into integer IDs.
input_ids = [char_to_id[ch] for ch in input_text]

# Convert each ID into a one-hot vector.
# Each character uses vocab_size numbers.
one_hot = np.eye(vocab_size)

# Join the three character vectors into one long vector.
x = one_hot[input_ids].flatten()

# Get the correct target ID.
target_id = char_to_id[target_char]

print("Vocabulary:", vocabulary)
print("Input IDs:", input_ids)
print("Input vector shape:", x.shape)
print("Correct target:", target_char)


# ------------------------------------------------------------
# PART 2: Initialize the model's parameters
# ------------------------------------------------------------

# This is our simple linear prediction model.
# It has one weight matrix and one bias vector.
W = np.random.randn(len(x), vocab_size) * 0.1
b = np.zeros(vocab_size)


# ------------------------------------------------------------
# PART 3: Define softmax
# ------------------------------------------------------------

def softmax(logits):
    # Subtract max(logits) to improve numerical stability.
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)

    # Normalize the scores so the probabilities sum to 1.
    return exp_values / exp_values.sum()


# ------------------------------------------------------------
# PART 4: Make a prediction and calculate the loss
# ------------------------------------------------------------

# Forward pass: calculate the output scores.
logits = x @ W + b

# Convert the scores into probabilities.
probabilities = softmax(logits)

# Loss depends on the probability of the correct target.
loss = -np.log(probabilities[target_id])

print("\nBEFORE UPDATE")
print("Logits:", np.round(logits, 4))
print("Probabilities:", np.round(probabilities, 4))
print("P(correct character):", round(probabilities[target_id], 4))
print("Loss:", round(loss, 4))


# ------------------------------------------------------------
# PART 5: Backpropagation
# ------------------------------------------------------------

# For softmax + cross-entropy, the gradient with respect
# to the logits is:
#
# gradient = predicted probabilities - target one-hot vector
#
# We start from the output and calculate how each output
# score contributes to the loss.

d_logits = probabilities.copy()
d_logits[target_id] -= 1.0

# Calculate gradients for the weight matrix and biases.
# np.outer combines the input features and output gradients.
dW = np.outer(x, d_logits)
db = d_logits

print("\nGradient of logits:", np.round(d_logits, 4))
print("Gradient shape for W:", dW.shape)
print("Gradient shape for b:", db.shape)


# ------------------------------------------------------------
# PART 6: Gradient descent
# ------------------------------------------------------------

learning_rate = 0.1

# Move each parameter in the direction that reduces the loss.
W -= learning_rate * dW
b -= learning_rate * db


# ------------------------------------------------------------
# PART 7: Check whether the model improved
# ------------------------------------------------------------

# Run another forward pass with the updated parameters.
new_logits = x @ W + b
new_probabilities = softmax(new_logits)
new_loss = -np.log(new_probabilities[target_id])

print("\nAFTER UPDATE")
print("Probabilities:", np.round(new_probabilities, 4))
print("P(correct character):", round(new_probabilities[target_id], 4))
print("Loss:", round(new_loss, 4))

# This should be true for this update and example.
assert new_loss < loss

print("\nSuccess! The loss decreased after the weight update.")

#================================================== TRAIN OUR TINY TEXT MODEL

# ============================================================
# STEP 7: TRAIN OUR TINY TEXT MODEL
#
# Concepts:
#   1. Dataset and sliding context windows
#   2. Forward pass
#   3. Softmax and cross-entropy loss
#   4. Backpropagation
#   5. Gradient descent
#   6. Epochs and training loss
#
# We use NumPy so that every calculation remains visible.
# ============================================================

import numpy as np

# ------------------------------------------------------------
# PART 1: Prepare the dataset
# ------------------------------------------------------------

# We deliberately use a very small, repeating text.
# This is a teaching dataset, not a realistic language corpus.
text = "hellohellohellohello"

# Build a vocabulary containing every unique character.
vocabulary = sorted(set(text))

# Assign an integer ID to each character.
char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

# Reverse mapping: integer ID -> character.
id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

# Number of unique characters the model can predict.
vocab_size = len(vocabulary)

print("Training text:", text)
print("Vocabulary:", vocabulary)
print("Vocabulary size:", vocab_size)


# ------------------------------------------------------------
# PART 2: Encode the text
# ------------------------------------------------------------

# Convert every character into its integer ID.
encoded = np.array([
    char_to_id[character]
    for character in text
])

print("\nEncoded text:")
print(encoded)


# ------------------------------------------------------------
# PART 3: Create input-target pairs
# ------------------------------------------------------------

# The model can look at three characters at a time.
context_length = 3

inputs = []
targets = []

# Slide a window across the text.
# Each input is followed by one character that we must predict.
for i in range(len(encoded) - context_length):

    # Three consecutive character IDs.
    context_ids = encoded[i:i + context_length]

    # The next character ID is the correct answer.
    target_id = encoded[i + context_length]

    inputs.append(context_ids)
    targets.append(target_id)

# These arrays contain all our training examples.
inputs = np.array(inputs)
targets = np.array(targets)

print("\nInput IDs shape:", inputs.shape)
print("Target IDs shape:", targets.shape)
print("Number of examples:", len(targets))


# ------------------------------------------------------------
# PART 4: One-hot encode the contexts
# ------------------------------------------------------------

# np.eye creates an identity matrix.
# Each row represents one character as a one-hot vector.
one_hot_table = np.eye(vocab_size)

# Convert each input character ID into its one-hot vector.
# Each example has shape (context_length, vocab_size).
encoded_inputs = one_hot_table[inputs]

# Flatten each context into one vector.
# Example: 3 characters x 4 values = 12 input features.
X = encoded_inputs.reshape(len(inputs), -1)

# Targets remain integer IDs because cross-entropy can use
# the correct class index directly.
Y = targets

print("\nX shape:", X.shape)
print("Y shape:", Y.shape)


# ------------------------------------------------------------
# PART 5: Initialize the model
# ------------------------------------------------------------

np.random.seed(42)

input_size = X.shape[1]

# W connects every input feature to every output character.
# Small random values give us a starting point for learning.
W = np.random.randn(input_size, vocab_size) * 0.1

# Each output character has its own bias.
b = np.zeros(vocab_size)

print("\nW shape:", W.shape)
print("b shape:", b.shape)


# ------------------------------------------------------------
# PART 6: Create a numerically stable softmax function
# ------------------------------------------------------------

def softmax(logits):
    # Work row by row because we predict for multiple examples.
    # Subtracting the row maximum helps prevent overflow.
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    exponentials = np.exp(shifted)

    # Normalize each row so its probabilities sum to 1.
    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )


# ------------------------------------------------------------
# PART 7: Forward pass and loss calculation
# ------------------------------------------------------------

def forward_and_loss(X, Y, W, b):

    # Forward pass:
    # Matrix multiplication produces one score per character.
    logits = X @ W + b

    # Convert scores into probabilities.
    probabilities = softmax(logits)

    # Select the probability assigned to each correct target.
    correct_probabilities = probabilities[
        np.arange(len(Y)), Y
    ]

    # Small epsilon protects log() against numerical zero.
    epsilon = 1e-12

    # Calculate the average cross-entropy loss.
    loss = -np.log(
        np.clip(correct_probabilities, epsilon, 1.0)
    ).mean()

    return logits, probabilities, loss


# ------------------------------------------------------------
# PART 8: Train the model
# ------------------------------------------------------------

learning_rate = 0.5
epochs = 500

print("\nTraining starts...")
print("Epoch | Average loss")

# An epoch means one pass through the entire training dataset.
for epoch in range(1, epochs + 1):

    # A. Forward pass: calculate predictions and loss.
    logits, probabilities, loss = forward_and_loss(
        X, Y, W, b
    )

    # B. Backpropagation:
    # For softmax + cross-entropy, the gradient with respect
    # to each logit is probability - correct_target.
    d_logits = probabilities.copy()

    # Subtract 1 from the probability of the correct class.
    d_logits[np.arange(len(Y)), Y] -= 1.0

    # Average the gradients across all training examples.
    d_logits /= len(Y)

    # C. Calculate parameter gradients.
    #
    # X.T @ d_logits combines input features with output
    # gradients to calculate the gradient of each weight.
    dW = X.T @ d_logits

    # Add together the output gradients to obtain bias gradients.
    db = np.sum(d_logits, axis=0)

    # D. Gradient descent:
    # Change each parameter in the opposite direction
    # of its gradient.
    W -= learning_rate * dW
    b -= learning_rate * db

    # E. Report the loss after the update at selected epochs.
    if epoch in [1, 10, 50, 100, 200, 500]:

        _, _, updated_loss = forward_and_loss(
            X, Y, W, b
        )

        print(f"{epoch:5} | {updated_loss:.4f}")


# ------------------------------------------------------------
# PART 9: Evaluate training predictions
# ------------------------------------------------------------

# Calculate the final predictions.
logits, probabilities, final_loss = forward_and_loss(
    X, Y, W, b
)

predicted_ids = np.argmax(probabilities, axis=1)

# Percentage of examples for which the most likely character
# matches the correct next character.
accuracy = np.mean(predicted_ids == Y)

print("\nTraining complete!")
print("Final loss:", round(final_loss, 4))
print("Training accuracy:", f"{accuracy * 100:.1f}%")


# ------------------------------------------------------------
# PART 10: Inspect some learned examples
# ------------------------------------------------------------

print("\nLearned predictions:")

for i in range(min(5, len(X))):

    context_text = "".join(
        id_to_char[int(token_id)]
        for token_id in inputs[i]
    )

    correct_char = id_to_char[int(Y[i])]
    predicted_char = id_to_char[int(predicted_ids[i])]

    print(
        f"Input: {context_text!r} | "
        f"Target: {correct_char!r} | "
        f"Prediction: {predicted_char!r}"
    )



