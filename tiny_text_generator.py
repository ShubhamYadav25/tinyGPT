
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



#============================================ Autoregressive text generation ===================
"""
Auto: the model uses its own previously generated output.
Regressive: it predicts the next part of a sequence from the preceding context.

we trained the model to predict the next character. But predicting one character isn't the same as generating a whole sequence. We need a way to use each prediction to make the next one.
That process is called autoregressive generation.

IDEA

Starting prompt
hel

Step 1: predict next character
l
Text becomes hell

Step 2: use the updated context
o
Text becomes hello

Keep repeating
hellohellohello...

The important detail is that the model doesn't predict the entire sentence at once. It predicts one character, adds that character to the sequence, and predicts again.

"""

# ============================================================
# STEP 8: AUTOREGRESSIVE TEXT GENERATION
#
# Goal:
#   Use our trained model to generate characters one by one.
#
# Important:
#   W and b must be the trained parameters from Step 7.
# ============================================================

# ------------------------------------------------------------
# PART 1: Choose a starting prompt
# ------------------------------------------------------------

# This is the initial text that we give to the model.
seed_text = "hel"

# How many new characters should the model generate?
max_new_chars = 15

# Greedy generation means we always select the character
# with the highest predicted probability.
#
# This makes our first experiment deterministic.
# We'll explore random sampling later.
generated_text = seed_text

print("\n" + "=" * 50)
print("AUTOREGRESSIVE TEXT GENERATION")
print("=" * 50)

print("Starting prompt:", repr(seed_text))
print()


# ------------------------------------------------------------
# PART 2: Generate one character at a time
# ------------------------------------------------------------

for step in range(1, max_new_chars + 1):

    # A. Take only the last context_length characters.
    #
    # For example:
    # generated_text = "hello"
    # context_length = 3
    # context_text = "llo"
    context_text = generated_text[-context_length:]

    # B. Convert the context characters into integer IDs.
    context_ids = [
        char_to_id[character]
        for character in context_text
    ]

    # C. Convert the IDs into one-hot vectors.
    # Each character becomes a vector of vocab_size values.
    context_vectors = one_hot_table[context_ids]

    # D. Flatten the vectors into the model's input format.
    # The batch dimension is 1 because we generate one example.
    x_next = context_vectors.flatten().reshape(1, -1)

    # E. Forward pass using the trained weights and biases.
    next_logits = x_next @ W + b

    # F. Convert the logits into probabilities.
    next_probabilities = softmax(next_logits)[0]

    # G. Choose the character with the highest probability.
    next_id = int(np.argmax(next_probabilities))
    next_char = id_to_char[next_id]

    # H. Display what happened in this generation step.
    confidence = next_probabilities[next_id]

    print(
        f"Step {step:2}: "
        f"context={context_text!r:5} | "
        f"next={next_char!r} | "
        f"probability={confidence:.4f}"
    )

    # I. Append the predicted character to our text.
    #
    # This is the key to autoregressive generation:
    # the model's own output becomes part of the next input.
    generated_text += next_char


# ------------------------------------------------------------
# PART 3: Display the completed sequence
# ------------------------------------------------------------

print("\nFinal generated text:")
print(generated_text)

print("\nOriginal prompt:", seed_text)
print("New characters generated:", len(generated_text) - len(seed_text))

"""
Greedy generation versus sampling
next_id = int(np.argmax(next_probabilities))

This is called greedy decoding: always select the most probable character.

Another approach is sampling:
next_id = int(np.random.choice(
    vocab_size,
    p=next_probabilities
))

Sampling chooses a character randomly according to the model's probability distribution. A less probable character can sometimes be selected.

"""

# ======================================== Embeddings and positional information ============================
"""
So far, our text generator uses one-hot vectors: each character is represented by a vector containing mostly zeros and one 1. That was useful for learning the fundamentals, but modern language models use a more flexible representation called an embedding.
In this:
1. Why token IDs aren't enough.
2. What embeddings are and how an embedding matrix works.
3. How positional information tells the model where each token occurs.
4. How to combine token embeddings and positional embeddings using NumPy.


Character	Token ID
e	0
h	1
l	2
o	3
A token ID is just a label. For example, the ID 3 for o doesn't mean that o is more important than e, whose ID is 0.

In our earlier model, we converted each ID into a one-hot vector. For example:
[0 1 0 0 0]

One-hot vectors work, but they don't give us a trainable, compact representation in which the model can learn useful features for each token.
An embedding is a vector of numbers that the model learns for each token.

For example, instead of representing l as [0, 0, 1, 0], our model might learn a four-dimensional vector such as:
l → [0.21, -0.43, 0.76, 0.12]

Positional information:

Sequence A: ell
Sequence B: lle

They contain the same characters, but the order is different.
Token embeddings alone tell us which character is present. They don't explicitly tell the model whether that character occurs at position 0, 1, or 2.

Transformers therefore need a way to represent order. In the version we're building, we'll use learned positional embeddings.
Each position gets its own vector, independently of the token at that position.

Token embedding (Which character is this?)

+
Position embedding (Where does it occur?)

Combined representation (Information about token identity and position)

ONE MORE EXAMPLE ON Positional embeddings:
- The dog bites the man.
- The man bites the dog.
The same words appear, but the order changes the meaning.
A language model needs to understand two things: which token it is looking at and where that token occurs in the sequence.

Notice that l appears twice: once at position 1 and once at position 2.
Both occurrences have the same token ID and token embedding, because they represent the same character.
But they occupy different positions.
"""

# ============================================================
# STEP 9: TOKEN EMBEDDINGS + POSITIONAL EMBEDDINGS
#
# Goal:
#   Represent each token as a vector and give it position info.
#
# Important:
#   These embeddings are randomly initialized for now.
#   They have NOT been trained to represent language patterns.
# ============================================================

import numpy as np

np.random.seed(42)


# ------------------------------------------------------------
# PART 1: Create the vocabulary
# ------------------------------------------------------------

text = "hello"

# Each unique character receives a stable integer ID.
vocabulary = sorted(set(text))

char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

vocab_size = len(vocabulary)

print("Vocabulary:", vocabulary)
print("Character IDs:", char_to_id)


# ------------------------------------------------------------
# PART 2: Define the embedding dimensions
# ------------------------------------------------------------

# How many numbers will represent each token?
embedding_dim = 4

# How many positions can the model accept at once?
context_length = 3

print("\nEmbedding dimension:", embedding_dim)
print("Maximum context length:", context_length)


# ------------------------------------------------------------
# PART 3: Initialize the token embedding matrix
# ------------------------------------------------------------

# One row for every vocabulary character.
# One column for every embedding feature.
#
# Shape: (vocab_size, embedding_dim)
#
# In a real model, these values are trainable parameters.
token_embeddings = (
    np.random.randn(vocab_size, embedding_dim) * 0.1
)

print("\nToken embedding matrix:")
print(np.round(token_embeddings, 3))

print("Token embedding shape:", token_embeddings.shape)


# ------------------------------------------------------------
# PART 4: Look up the embeddings for "ell"
# ------------------------------------------------------------

input_text = "ell"

# Convert characters into integer IDs.
token_ids = [
    char_to_id[character]
    for character in input_text
]

# Each ID selects one row from the embedding matrix.
# The result has one row per token.
token_vectors = token_embeddings[token_ids]

print("\nInput text:", input_text)
print("Token IDs:", token_ids)

print("\nToken vectors:")
print(np.round(token_vectors, 3))

print("Token vectors shape:", token_vectors.shape)


# ------------------------------------------------------------
# PART 5: Initialize positional embeddings
# ------------------------------------------------------------

# Every position gets a vector of embedding_dim numbers.
#
# Shape: (context_length, embedding_dim)
#
# These are separate, trainable parameters.
position_embeddings = (
    np.random.randn(context_length, embedding_dim) * 0.1
)

print("\nPositional embedding matrix:")
print(np.round(position_embeddings, 3))

print(
    "Position embedding shape:",
    position_embeddings.shape
)


# ------------------------------------------------------------
# PART 6: Select positions for our sequence
# ------------------------------------------------------------

# "ell" has three characters at positions 0, 1, and 2.
positions = np.arange(len(token_ids))

# Select the positional vector for each position.
position_vectors = position_embeddings[positions]

print("\nPosition vectors:")
print(np.round(position_vectors, 3))


# ------------------------------------------------------------
# PART 7: Combine token and position information
# ------------------------------------------------------------

# Add corresponding elements of both matrices.
#
# token_vectors:    (3, 4)
# position_vectors: (3, 4)
# combined:         (3, 4)

combined_embeddings = token_vectors + position_vectors

print("\nCombined embeddings:")
print(np.round(combined_embeddings, 3))

print(
    "Combined embedding shape:",
    combined_embeddings.shape
)


# ------------------------------------------------------------
# PART 8: Verify the effect of position
# ------------------------------------------------------------

# In "ell", the second and third characters are both 'l'.
# Their token embeddings should be identical.
same_token = np.allclose(
    token_vectors[1],
    token_vectors[2]
)

# But they occur at different positions, so their combined
# representations should differ for these random values.
different_positions = not np.allclose(
    combined_embeddings[1],
    combined_embeddings[2]
)

print("\nAre the two 'l' token embeddings equal?", same_token)
print(
    "Are their combined representations different?",
    different_positions
)

assert same_token
assert different_positions

print("\nEmbedding and position checks passed!")

#============================================ Q, K, V ============================================
"""
The weight matrices \(W_Q\), \(W_K\), and \(W_V\) contain learnable parameters. During training, the model adjusts them to produce useful query, key, and value representations.

Matrix	Shape	Meaning
\(X\)	(3, 4)	3 input tokens, 4 features each
\(W_Q\)	(4, 3)	Projection into query space
\(W_K\)	(4, 3)	Projection into key space
\(W_V\)	(4, 3)	Projection into value space
\(Q, K, V\)	(3, 3) each	One 3-number vector per token

Why do the shapes work? Matrix multiplication follows this rule:
(3, 4) @ (4, 3) = (3, 3)

"""


# ============================================================
# STEP 10: QUERY, KEY, AND VALUE
#
# Goal:
#   Calculate Q, K, and V from token + position embeddings.
#
# We use NumPy to expose the matrix calculations.
# These parameters are randomly initialized, not trained.
# ============================================================

import numpy as np

np.random.seed(42)

# ------------------------------------------------------------
# PART 1: Create our vocabulary and input sequence
# ------------------------------------------------------------

text = "hello"

# Each unique character gets one stable token ID.
vocabulary = sorted(set(text))

char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

input_text = "hel"

# Convert the input text to integer IDs.
token_ids = [
    char_to_id[character]
    for character in input_text
]

print("Vocabulary:", vocabulary)
print("Input text:", input_text)
print("Token IDs:", token_ids)


# ------------------------------------------------------------
# PART 2: Configure the embedding dimensions
# ------------------------------------------------------------

vocab_size = len(vocabulary)

# Each character gets a 4-number embedding vector.
d_model = 4

# Our sequence contains three characters.
sequence_length = len(token_ids)

# Each Q, K, and V vector will have 3 numbers.
d_k = 3
d_v = 3


# ------------------------------------------------------------
# PART 3: Create token and position embeddings
# ------------------------------------------------------------

# Token embedding matrix:
# one row per vocabulary character.
token_embedding_table = (
    np.random.randn(vocab_size, d_model) * 0.1
)

# Position embedding matrix:
# one row per position in our input sequence.
position_embedding_table = (
    np.random.randn(sequence_length, d_model) * 0.1
)

# Look up the vector for each input character.
token_vectors = token_embedding_table[token_ids]

# Select the vector for each position.
position_vectors = position_embedding_table[
    np.arange(sequence_length)
]

# Combine token identity and positional information.
# Shape: (sequence_length, d_model) = (3, 4)
X = token_vectors + position_vectors

print("\nInput representation X:")
print(np.round(X, 3))
print("X shape:", X.shape)


# ------------------------------------------------------------
# PART 4: Initialize the three projection matrices
# ------------------------------------------------------------

# Each projection transforms 4 input features into 3 features.
#
# W_Q: used to calculate queries.
# W_K: used to calculate keys.
# W_V: used to calculate values.

W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

print("\nW_Q shape:", W_Q.shape)
print("W_K shape:", W_K.shape)
print("W_V shape:", W_V.shape)


# ------------------------------------------------------------
# PART 5: Calculate Q, K, and V
# ------------------------------------------------------------

# Every row of X is transformed using each weight matrix.
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("\nQuery matrix Q:")
print(np.round(Q, 3))
print("Q shape:", Q.shape)

print("\nKey matrix K:")
print(np.round(K, 3))
print("K shape:", K.shape)

print("\nValue matrix V:")
print(np.round(V, 3))
print("V shape:", V.shape)


# ------------------------------------------------------------
# PART 6: Inspect the vectors for each character
# ------------------------------------------------------------

print("\nQ, K, and V for each token:")

for i, character in enumerate(input_text):
    print(f"\nToken: {character!r} at position {i}")
    print("  Query:", np.round(Q[i], 3))
    print("  Key:  ", np.round(K[i], 3))
    print("  Value:", np.round(V[i], 3))


# ------------------------------------------------------------
# PART 7: Verify the dimensions
# ------------------------------------------------------------

assert X.shape == (3, 4)
assert Q.shape == (3, 3)
assert K.shape == (3, 3)
assert V.shape == (3, 3)

print("\nAll matrix shape checks passed!")

"""
What do we actually do with Q, K, and V?

This is where the pieces connect.
Suppose the model is processing the character l in hel. It needs to determine how much information to take from the other characters: h, e, and l.
It uses the query for l to compare against the keys of all three tokens.
The comparison generates attention scores:

1. Compare Queries and Keys
Calculate how relevant each token is to another.

2. Softmax the scores
Turn each token's scores into attention weights.

3. Combine the Values
Use those weights to mix information from the value vectors.

In other words:
- Q and K determine where attention goes.
- V determines what information is gathered.
The weights are not manually programmed meanings. The model learns useful projections during training.
"""

"""
Imagine our input is: the cat sat
Suppose the model is processing the token sat. To understand this token in context, it may benefit from information about cat, because the cat is the one doing the sitting.

A useful language model needs a way to decide:
1. Which other tokens should I pay attention to?
2. How relevant is each token to me?
3. What information should I take from those tokens?

That's the problem self-attention solves.
Without attention, a model can still learn patterns, but it has no explicit mechanism for dynamically calculating relationships between tokens and mixing their information according to those relationships.

Q, K, and V

Q
Query: What information am I looking for?
The token uses its query to search for relevant information among the other tokens.



K
Key: How can other tokens recognize my relevance?
Each token has a key that gets compared with the query to calculate how well they match.



V
Value: What information can I contribute?
This is the representation whose information gets combined after the model calculates the attention weights.

The distinction matters:
- Q and K help decide where attention should go.
- V contains what information is passed along.

The comparison produces scores. Higher relative scores lead to larger attention weights. Those weights determine how much each word's value contributes to the resulting representation of sat.
This is what lets a Transformer build context-aware representations of tokens.

We create three separate matrices:
Matrix	What it learns to produce
\(W_Q\)	Query features for searching for relevant tokens
\(W_K\)	Key features for matching incoming queries
\(W_V\)	Value features for passing useful information along

Think of the three matrices as three different filters applied to the same input.
The input is the same, but each filter transforms it differently.

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

Each equation says:
"Take every token's input vector and transform it using the matrix for this particular role."

Starting input vector
0.2 0.5 -0.1 0.7

4 input features

Apply three different transformations
Q
Using WQ
Search features


K
Using WK
Matching features


V
Using WV
Information features

Why is Q, K, and V calculated for every token?

Because every token may need to gather information from other tokens.
Notice something important: the weight matrices are shared across all the tokens.

We don't create a separate \(W_Q\) for h, another for e, and another for l. We use the same \(W_Q\) for every token, and the different input vectors produce different queries.
That's an important property of Transformers: the same learned projection rules are applied at every position.

What do we actually achieve by calculating Q, K, and V?

Let's follow the complete flow.
Input X
Token embeddings + positional information

Create Q, K, V
Learned projections of each token

Compare Q with K
Calculate a score for each token pair

Softmax + causal mask
Convert allowed scores into attention weights

Combine V using the attention weights
Create context-aware token representations

First: Q and K determine attention weights
The model compares each query with every key using a dot product:
This produces an attention score matrix \(S\).

For three input tokens, the matrix has shape (3, 3). Each row represents the token doing the attending; each column represents a token that could supply information.
After applying the causal mask and softmax, the result is an attention-weight matrix \(A\).

Second: attention weights determine how V is combined
The model calculates:  O=AV

Suppose a token gives 70% attention weight to one other token and 30% to another. Its resulting representation will combine 70% of the first value vector and 30% of the second, along with any contribution from other permitted positions

Concept	Purpose
Q, K, V	Represent tokens for matching and information transfer
Attention weights	Decide how much information to gather from each token
Attention output	The new context-aware representation created from the value vectors

What is the causal mask, and why does GPT need it?
This is one of the most important details for a text generator.
Suppose you're generating a sentence from left to right. When predicting the next character after he, the model must not be allowed to look at future characters that haven't been generated yet.
For GPT, token 1 can attend only to token 1. Token 2 can attend to tokens 1 and 2. Token 3 can attend to tokens 1, 2, and 3.
"""

# ============================================================
# STEP 11: SINGLE-HEAD CAUSAL SELF-ATTENTION
#
# Goal:
#   1. Calculate query-key attention scores.
#   2. Block access to future tokens.
#   3. Convert scores into attention weights using softmax.
#   4. Combine value vectors using those weights.
#
# These Q, K, and V values are chosen for learning.
# They are not outputs of a trained neural network.
# ============================================================

import numpy as np

np.set_printoptions(precision=3, suppress=True)


# ------------------------------------------------------------
# PART 1: Define Q, K, and V
# ------------------------------------------------------------

# Three tokens, two features in each query vector.
Q = np.array([
    [1.0, 0.0],   # Query for token 1
    [0.0, 1.0],   # Query for token 2
    [1.0, 1.0],   # Query for token 3
])

# Keys have the same feature dimension as queries.
# We use the same numbers as Q for simpler arithmetic.
K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

# Each token has a two-dimensional value vector.
# These values carry the information to be combined.
V = np.array([
    [10.0, 1.0],  # Value of token 1
    [2.0, 8.0],   # Value of token 2
    [4.0, 3.0],   # Value of token 3
])

sequence_length = Q.shape[0]
d_k = Q.shape[1]

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# ------------------------------------------------------------
# PART 2: Calculate attention scores
# ------------------------------------------------------------

# K.T transposes the key matrix.
#
# Q @ K.T compares each query with every key.
# Dividing by sqrt(d_k) scales the scores.
#
# Result shape: (number of queries, number of keys)
scores = (Q @ K.T) / np.sqrt(d_k)

print("\nRaw attention scores:")
print(scores)


# ------------------------------------------------------------
# PART 3: Build the causal mask
# ------------------------------------------------------------

# np.triu creates an upper-triangular matrix.
# k=1 selects positions strictly above the diagonal.
#
# Those positions represent future tokens.
#
# True  = future position, must be blocked.
# False = current or previous position, allowed.

future_mask = np.triu(
    np.ones(
        (sequence_length, sequence_length),
        dtype=bool
    ),
    k=1
)

print("\nFuture-position mask:")
print(future_mask)


# ------------------------------------------------------------
# PART 4: Block future positions
# ------------------------------------------------------------

# Make a copy so we preserve the original scores.
masked_scores = scores.copy()

# Negative infinity ensures future positions receive
# exactly zero weight after softmax.
masked_scores[future_mask] = -np.inf

print("\nMasked attention scores:")
print(masked_scores)


# ------------------------------------------------------------
# PART 5: Implement row-wise softmax
# ------------------------------------------------------------

def softmax_rows(matrix):

    # Subtract the maximum from every row for numerical stability.
    shifted = matrix - np.max(
        matrix,
        axis=1,
        keepdims=True
    )

    # Exponentiate the shifted scores.
    exponentials = np.exp(shifted)

    # Normalize each row independently.
    # The rows will sum to 1.
    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )


# Convert masked scores into attention weights.
attention_weights = softmax_rows(masked_scores)

print("\nAttention weights:")
print(attention_weights)

print("\nSum of each attention row:")
print(attention_weights.sum(axis=1))


# ------------------------------------------------------------
# PART 6: Combine the value vectors
# ------------------------------------------------------------

# Each output row is a weighted combination of values.
#
# Output shape:
# (sequence_length, value_dimension)
output = attention_weights @ V

print("\nAttention output:")
print(output)

print("Output shape:", output.shape)


# ------------------------------------------------------------
# PART 7: Verify important properties
# ------------------------------------------------------------

# Every row must sum to approximately 1.
assert np.allclose(
    attention_weights.sum(axis=1),
    np.ones(sequence_length)
)

# No token should attend to a future token.
assert np.allclose(
    np.triu(attention_weights, k=1),
    0.0
)

# One output vector must exist for every input token.
assert output.shape == V.shape

print("\nAll attention checks passed!")

"""
Why do we call it “single-head” attention?

The word head means one independent attention mechanism.
Imagine we have the input h e l. Each character has its own vector. One attention head transforms those vectors into Q, K, and V, calculates one attention-weight matrix, and uses it to combine the values.

Single-head attention
Input X
All token vectors

One set of projections
Q, K, V

One attention-weight matrix
One distribution for each token

One attention output

What changes in multi-head attention?
Multi-head attention runs multiple independent attention heads in parallel.

Same input X

Head 1
Q₁, K₁, V₁

Attention output 1


Head 2
Q₂, K₂, V₂

Attention output 2



Combine head outputs
Usually concatenate them, then apply an output projection.

Each head has its own learned projection matrices. One head might learn useful relationships between nearby tokens, while another might learn different relationships. These are possibilities, not manually assigned jobs; training determines what each head learns.

Remember: single-head and multi-head refer to the number of attention mechanisms, not the number of tokens or dimensions.

"""


"""
Input text
"hello"

Token + positional embeddings
Create X automatically

Learnable projection matrices
Initialize WQ, WK, WV

Calculate Q, K, V
Matrix multiplication

Causal self-attention
Scores → mask → softmax → weighted values

Attention output
A new representation for every token
"""

# ============================================================
# STEP 11: COMPLETE SINGLE-HEAD CAUSAL SELF-ATTENTION
#
# No hardcoded Q, K, or V values!
#
# Everything is calculated from:
#   1. Input text
#   2. Token embeddings
#   3. Positional embeddings
#   4. Automatically initialized projection matrices
#
# Note:
#   This demonstrates a forward pass.
#   The parameters are NOT trained yet.
# ============================================================

import numpy as np

# Fix the random seed so the example is reproducible.
# Running the code again will produce the same initial values.
rng = np.random.default_rng(42)

np.set_printoptions(precision=3, suppress=True)


# ------------------------------------------------------------
# PART 1: Convert text into token IDs
# ------------------------------------------------------------

# Our tiny example text.
# We are using characters as tokens to keep things simple.
training_text = "hello"

# Extract the unique characters and sort them.
vocabulary = sorted(set(training_text))

# Create both mappings automatically.
char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

# Choose an input sequence.
input_text = "hello"

# Convert each input character to an integer ID.
token_ids = np.array([
    char_to_id[character]
    for character in input_text
])

vocab_size = len(vocabulary)
sequence_length = len(token_ids)

print("Vocabulary:", vocabulary)
print("Character IDs:", char_to_id)
print("Input text:", input_text)
print("Token IDs:", token_ids)


# ------------------------------------------------------------
# PART 2: Set the dimensions
# ------------------------------------------------------------

# d_model: number of features in each input embedding.
d_model = 4

# d_k: number of features in each query and key.
d_k = 3

# d_v: number of features in each value vector.
d_v = 3

print("\nInput settings:")
print("Vocabulary size:", vocab_size)
print("Sequence length:", sequence_length)
print("Model dimension:", d_model)
print("Query/key dimension:", d_k)
print("Value dimension:", d_v)


# ------------------------------------------------------------
# PART 3: Initialize token embeddings
# ------------------------------------------------------------

# Each vocabulary character gets a vector of d_model numbers.
#
# Shape: (vocab_size, d_model)
#
# These are random initial parameters.
# Training will eventually adjust them.

token_embedding_table = rng.normal(
    loc=0.0,
    scale=0.1,
    size=(vocab_size, d_model)
)

print("\nToken embedding table shape:",
      token_embedding_table.shape)

# Look up the embedding for every token in our input.
#
# Example:
# If token ID 1 represents 'h', use row 1 of the table.
token_vectors = token_embedding_table[token_ids]


# ------------------------------------------------------------
# PART 4: Initialize positional embeddings
# ------------------------------------------------------------

# Each position gets a vector of d_model numbers.
#
# For this small demonstration, the table has one row
# for every position in the input sequence.
#
# Shape: (sequence_length, d_model)

position_embedding_table = rng.normal(
    loc=0.0,
    scale=0.1,
    size=(sequence_length, d_model)
)

print("Position embedding table shape:",
      position_embedding_table.shape)

# Position 0, position 1, and so on.
positions = np.arange(sequence_length)

# Select the position vector for each token.
position_vectors = position_embedding_table[positions]


# ------------------------------------------------------------
# PART 5: Construct X
# ------------------------------------------------------------

# Add the token and position vectors.
#
# token_vectors:    (sequence_length, d_model)
# position_vectors: (sequence_length, d_model)
# X:                (sequence_length, d_model)

X = token_vectors + position_vectors

print("\nInput matrix X:")
print(X)
print("X shape:", X.shape)


# ------------------------------------------------------------
# PART 6: Initialize the projection matrices
# ------------------------------------------------------------

# The model needs three separate transformations.
#
# W_Q produces queries.
# W_K produces keys.
# W_V produces values.
#
# These are automatically initialized, not manually specified.
#
# W_Q: (d_model, d_k)
# W_K: (d_model, d_k)
# W_V: (d_model, d_v)

W_Q = rng.normal(
    loc=0.0,
    scale=1 / np.sqrt(d_model),
    size=(d_model, d_k)
)

W_K = rng.normal(
    loc=0.0,
    scale=1 / np.sqrt(d_model),
    size=(d_model, d_k)
)

W_V = rng.normal(
    loc=0.0,
    scale=1 / np.sqrt(d_model),
    size=(d_model, d_v)
)

print("\nW_Q shape:", W_Q.shape)
print("W_K shape:", W_K.shape)
print("W_V shape:", W_V.shape)


# ------------------------------------------------------------
# PART 7: Calculate Q, K, and V
# ------------------------------------------------------------

# Each token vector is transformed using the same
# projection matrix for that particular role.
#
# Q, K and V are calculated for EVERY token.
#
# Q = X @ W_Q
# K = X @ W_K
# V = X @ W_V

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("\nQ matrix:")
print(Q)
print("Q shape:", Q.shape)

print("\nK matrix:")
print(K)
print("K shape:", K.shape)

print("\nV matrix:")
print(V)
print("V shape:", V.shape)


# ------------------------------------------------------------
# PART 8: Calculate query-key attention scores
# ------------------------------------------------------------

# Transpose K so its features align with Q for multiplication.
#
# Q:   (sequence_length, d_k)
# K.T: (d_k, sequence_length)
#
# Result: (sequence_length, sequence_length)

scores = (Q @ K.T) / np.sqrt(d_k)

print("\nAttention scores before masking:")
print(scores)
print("Scores shape:", scores.shape)


# ------------------------------------------------------------
# PART 9: Apply the causal mask
# ------------------------------------------------------------

# Each token may attend to itself and previous tokens.
# It must not attend to tokens occurring later in the sequence.
#
# np.triu(..., k=1) identifies positions strictly above
# the main diagonal: the future positions.

future_mask = np.triu(
    np.ones(
        (sequence_length, sequence_length),
        dtype=bool
    ),
    k=1
)

# Replace future-position scores with negative infinity.
masked_scores = scores.copy()
masked_scores[future_mask] = -np.inf

print("\nScores after causal masking:")
print(masked_scores)


# ------------------------------------------------------------
# PART 10: Convert scores into attention weights
# ------------------------------------------------------------

def softmax_rows(matrix):
    """
    Apply softmax independently to every row.

    Subtracting each row's maximum makes the calculation
    more numerically stable.
    """

    row_max = np.max(
        matrix,
        axis=1,
        keepdims=True
    )

    shifted = matrix - row_max
    exponentials = np.exp(shifted)

    probabilities = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )

    return probabilities


# Future positions have -inf scores, so softmax assigns
# them zero attention weight.
attention_weights = softmax_rows(masked_scores)

print("\nAttention weights:")
print(attention_weights)

print("\nSum of each attention row:")
print(attention_weights.sum(axis=1))


# ------------------------------------------------------------
# PART 11: Combine the value vectors
# ------------------------------------------------------------

# Multiply attention weights by V.
#
# Each output row is a weighted combination of the allowed
# value vectors for that token.
#
# attention_weights: (sequence_length, sequence_length)
# V:                 (sequence_length, d_v)
# output:            (sequence_length, d_v)

output = attention_weights @ V

print("\nAttention output:")
print(output)
print("Output shape:", output.shape)


# ------------------------------------------------------------
# PART 12: Verify the results
# ------------------------------------------------------------

# Every attention row should sum to 1.
assert np.allclose(
    attention_weights.sum(axis=1),
    np.ones(sequence_length)
)

# No token should attend to a future position.
assert np.allclose(
    np.triu(attention_weights, k=1),
    0.0
)

# We should have one output vector per input token.
assert output.shape == (sequence_length, d_v)

print("\nAll checks passed!")
print("The attention forward pass is complete.")

"""
Attention output:
[[ 0.053 -0.054 -0.039]
 [-0.024 -0.106 -0.033]
 [-0.027 -0.118 -0.04 ]
 [-0.046 -0.117 -0.007]
 [-0.032 -0.099  0.003]]
 
 Our input was hello, which has five character positions:
Row	Input character	What this output represents
0	h	New representation of h
1	e	New representation of e, using allowed context
2	l	New representation of l, using allowed context
3	l	New representation of the second l, using allowed context
4	o	New representation of o, using allowed context

This is the new representation of the first token, h. => [ 0.053 -0.054 -0.039]

The attention weights tell us how much each allowed token's value contributes to a particular token's new representation
"""

"""
Why do we need multiple heads?

One attention head calculates one set of attention weights. Another head calculates a different set using its own projection matrices.
Conceptually, different heads can learn different relationships between tokens. For example, one might become useful for local patterns, while another might capture a different contextual relationship. These are possibilities learned during training, not fixed jobs assigned to the heads.

Notice that each head uses four features because we have chosen d_model = 8 and num_heads = 2.
\[
d_{\text{head}}=\frac{d_{\text{model}}}{\text{num heads}}
=\frac{8}{2}=4
\]


"""


# ============================================================
# STEP 12: MULTI-HEAD CAUSAL SELF-ATTENTION
#
# No manually entered Q, K, or V values.
#
# Each head has its own randomly initialized projection matrices.
# The input is calculated from token + position embeddings.
#
# This is a forward pass, not a trained Transformer yet.
# ============================================================

import numpy as np

rng = np.random.default_rng(42)
np.set_printoptions(precision=3, suppress=True)


# ------------------------------------------------------------
# PART 1: Prepare the input text
# ------------------------------------------------------------

text = "hello"

# Create the vocabulary automatically.
vocabulary = sorted(set(text))

# Character -> integer ID.
char_to_id = {
    char: index
    for index, char in enumerate(vocabulary)
}

# Integer ID -> character.
id_to_char = {
    index: char
    for char, index in char_to_id.items()
}

# Encode the input text.
token_ids = np.array([
    char_to_id[char]
    for char in text
])

vocab_size = len(vocabulary)
sequence_length = len(token_ids)

print("Text:", text)
print("Vocabulary:", vocabulary)
print("Token IDs:", token_ids)


# ------------------------------------------------------------
# PART 2: Configure the model dimensions
# ------------------------------------------------------------

# Number of features representing each token.
d_model = 8

# Number of independent attention heads.
num_heads = 2

# For this model, divide model features evenly across heads.
assert d_model % num_heads == 0

head_dim = d_model // num_heads

print("\nModel dimension:", d_model)
print("Number of heads:", num_heads)
print("Dimension per head:", head_dim)


# ------------------------------------------------------------
# PART 3: Create token embeddings
# ------------------------------------------------------------

# Each vocabulary token gets a vector of d_model values.
#
# Shape: (vocab_size, d_model)
token_embedding_table = rng.normal(
    loc=0.0,
    scale=0.1,
    size=(vocab_size, d_model)
)

# Select one embedding row for each input token.
token_vectors = token_embedding_table[token_ids]

print("\nToken embedding shape:", token_vectors.shape)


# ------------------------------------------------------------
# PART 4: Create positional embeddings
# ------------------------------------------------------------

# Each position gets its own d_model-dimensional vector.
#
# Shape: (sequence_length, d_model)
position_embedding_table = rng.normal(
    loc=0.0,
    scale=0.1,
    size=(sequence_length, d_model)
)

# Select the vector belonging to each position.
positions = np.arange(sequence_length)
position_vectors = position_embedding_table[positions]

# Combine token identity and position.
#
# X shape: (sequence_length, d_model)
X = token_vectors + position_vectors

print("Input X shape:", X.shape)


# ------------------------------------------------------------
# PART 5: Initialize separate projection matrices per head
# ------------------------------------------------------------

# Each attention head gets its own W_Q, W_K, and W_V.
#
# Every matrix is initialized automatically from the random
# number generator. None of Q, K, or V is manually specified.
#
# Each projection matrix has shape: (d_model, head_dim).

W_Q = [
    rng.normal(
        loc=0.0,
        scale=1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]

W_K = [
    rng.normal(
        loc=0.0,
        scale=1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]

W_V = [
    rng.normal(
        loc=0.0,
        scale=1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]


# ------------------------------------------------------------
# PART 6: Implement causal attention
# ------------------------------------------------------------

def causal_attention(Q, K, V):
    """
    Calculate attention for one head.

    Q, K: (sequence_length, head_dim)
    V:    (sequence_length, head_dim)

    Returns:
        output:  (sequence_length, head_dim)
        weights: (sequence_length, sequence_length)
    """

    # Compare every query with every key.
    # Scaling helps keep dot-product scores well behaved.
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])

    # Block future positions.
    # Rows represent query positions; columns represent keys.
    future_mask = np.triu(
        np.ones(scores.shape, dtype=bool),
        k=1
    )

    scores[future_mask] = -np.inf

    # Row-wise softmax, with a stable maximum subtraction.
    shifted_scores = scores - np.max(
        scores,
        axis=1,
        keepdims=True
    )

    exponentials = np.exp(shifted_scores)

    weights = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )

    # Combine value vectors using the attention weights.
    output = weights @ V

    return output, weights


# ------------------------------------------------------------
# PART 7: Run each attention head
# ------------------------------------------------------------

head_outputs = []
all_attention_weights = []

for head_index in range(num_heads):

    print(f"\n{'=' * 15} HEAD {head_index + 1} {'=' * 15}")

    # Calculate this head's Q, K and V from the SAME input X.
    # The projections differ because each head has its own weights.
    Q = X @ W_Q[head_index]
    K = X @ W_K[head_index]
    V = X @ W_V[head_index]

    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)

    # Calculate the output and attention distribution.
    head_output, attention_weights = causal_attention(
        Q, K, V
    )

    print("\nAttention weights:")
    print(attention_weights)

    print("\nThis head's output:")
    print(head_output)

    head_outputs.append(head_output)
    all_attention_weights.append(attention_weights)


# ------------------------------------------------------------
# PART 8: Concatenate the head outputs
# ------------------------------------------------------------

# Each head output has shape (sequence_length, head_dim).
#
# Concatenating along axis=1 joins their feature columns.
#
# With two heads:
# (5, 4) + (5, 4) -> (5, 8)

concatenated = np.concatenate(
    head_outputs,
    axis=1
)

print("\nConcatenated output:")
print(concatenated)
print("Concatenated shape:", concatenated.shape)


# ------------------------------------------------------------
# PART 9: Apply the output projection
# ------------------------------------------------------------

# The output projection combines the features from all heads.
#
# Shape:
# (num_heads * head_dim, d_model)
#
# Here: (8, 8)

W_O = rng.normal(
    loc=0.0,
    scale=1 / np.sqrt(d_model),
    size=(num_heads * head_dim, d_model)
)

# Project the joined features back to d_model dimensions.
final_output = concatenated @ W_O

print("\nFinal multi-head attention output:")
print(final_output)
print("Final output shape:", final_output.shape)


# ------------------------------------------------------------
# PART 10: Validate the result
# ------------------------------------------------------------

# Each head's attention probabilities must sum to one per row.
for weights in all_attention_weights:
    assert np.allclose(
        weights.sum(axis=1),
        np.ones(sequence_length)
    )

    # No head may attend to future positions.
    assert np.allclose(
        np.triu(weights, k=1),
        0.0
    )

# The final output keeps the original model feature dimension.
assert final_output.shape == (sequence_length, d_model)

print("\nAll multi-head attention checks passed!")

#=============================================================================================
"""
Transformer block

multi-head attention is only one part of a Transformer. A Transformer block adds other components that help the model process and refine the information it has gathered.

What components are inside a Transformer block?

Input X
Token and positional embeddings

1. Layer normalization
Normalize each token's feature values

2. Multi-head causal self-attention
Let each token gather information from earlier tokens and itself

Add the original input using a residual connection

3. Layer normalization
Normalize the updated token representations

4. Feed-forward network
Transform each token's features using a small neural network

Add another residual connection

Transformer block output
Updated representation for every token


A. Residual connections — why add the original input back?
Suppose a token enters the attention layer with a vector \(X\). Attention calculates an updated vector \(A(X)\).
A residual connection combines them:
\[
H=X+A(X)
\]

Why do this?
Attention can add useful contextual information, but we don't necessarily want to discard the existing representation. The residual connection provides a direct path for the original information to continue through the network.

B. Layer normalization — why normalize?
The feature values in a neural network can have different scales as information flows through layers.
Layer normalization standardizes the features of each token, then applies learnable scale and shift parameters.
Conceptually, it helps keep the representations in a better-behaved numerical range during training.
Unlike normalizing an entire dataset, layer normalization operates across the feature dimensions of an individual token. For our input matrix of shape (5, 8), it normalizes each row across its eight features.
C. Feed-forward network — what does attention not do?
Attention gathers and mixes information from different token positions. The feed-forward network performs a different job: it transforms the features of each token independently.
We'll use two linear transformations with a ReLU activation:
\[
\operatorname{FFN}(x)=
\operatorname{ReLU}(xW_1+b_1)W_2+b_2
\]

The first layer expands the feature dimension. The second returns it to the model dimension.
For our example, we'll go from 8 features to 16, then back to 8.
The same feed-forward weights are applied at every token position, but attention weights determine how information moves between positions.

"""


# ============================================================
# STEP 13: BUILD ONE COMPLETE TRANSFORMER BLOCK
#
# Architecture:
#   Input
#     -> LayerNorm
#     -> Multi-head causal self-attention
#     -> Residual connection
#     -> LayerNorm
#     -> Feed-forward network
#     -> Residual connection
#     -> Output
#
# No hardcoded Q, K, or V values.
# All projection matrices are initialized automatically.
#
# Important:
#   This demonstrates a forward pass, not a trained GPT.
# ============================================================

import numpy as np

rng = np.random.default_rng(42)
np.set_printoptions(precision=3, suppress=True)


# ------------------------------------------------------------
# PART 1: Prepare the input text and vocabulary
# ------------------------------------------------------------

text = "hello"

# Create unique character IDs automatically.
vocabulary = sorted(set(text))

char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

# Encode every character as its integer ID.
token_ids = np.array([
    char_to_id[character]
    for character in text
])

vocab_size = len(vocabulary)
sequence_length = len(token_ids)

print("Input text:", text)
print("Vocabulary:", vocabulary)
print("Token IDs:", token_ids)


# ------------------------------------------------------------
# PART 2: Configure the Transformer block
# ------------------------------------------------------------

# Each token will be represented by eight features.
d_model = 8

# Use two independent attention heads.
num_heads = 2

# Each head will create four query, key, and value features.
assert d_model % num_heads == 0
head_dim = d_model // num_heads

# The feed-forward network expands 8 features to 16.
hidden_dim = 16

print("\nModel dimension:", d_model)
print("Number of attention heads:", num_heads)
print("Dimension per head:", head_dim)
print("Feed-forward hidden dimension:", hidden_dim)


# ------------------------------------------------------------
# PART 3: Create token and positional embeddings
# ------------------------------------------------------------

# One learned vector per character in the vocabulary.
# Shape: (vocab_size, d_model)
token_embedding_table = rng.normal(
    0.0, 0.1, size=(vocab_size, d_model)
)

# One learned vector per position in this example.
# Shape: (sequence_length, d_model)
position_embedding_table = rng.normal(
    0.0, 0.1, size=(sequence_length, d_model)
)

# Look up the input token embeddings.
token_vectors = token_embedding_table[token_ids]

# Look up the positional embeddings.
positions = np.arange(sequence_length)
position_vectors = position_embedding_table[positions]

# Combine token identity with position information.
X = token_vectors + position_vectors

print("\nInput X shape:", X.shape)


# ------------------------------------------------------------
# PART 4: Implement layer normalization
# ------------------------------------------------------------

def layer_norm(x, gamma, beta, epsilon=1e-5):
    """
    Normalize each token independently across its features.

    x shape: (..., d_model)
    gamma and beta shape: (d_model,)

    Gamma controls scale; beta controls shift.
    Both can be learned during training.
    """

    # Calculate the mean of each token's feature vector.
    mean = np.mean(x, axis=-1, keepdims=True)

    # Calculate the variance of each token's feature vector.
    variance = np.var(x, axis=-1, keepdims=True)

    # Standardize the features.
    normalized = (x - mean) / np.sqrt(variance + epsilon)

    # Apply the learnable scale and shift parameters.
    return gamma * normalized + beta


# Layer normalization starts with scale=1 and shift=0.
# Training can adjust these parameters later.
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)

gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)


# ------------------------------------------------------------
# PART 5: Initialize attention projection matrices
# ------------------------------------------------------------

# Each head gets its own W_Q, W_K, and W_V.
#
# The model learns these matrices during training.
# For now, they are random initial parameters.

W_Q = [
    rng.normal(
        0.0, 1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]

W_K = [
    rng.normal(
        0.0, 1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]

W_V = [
    rng.normal(
        0.0, 1 / np.sqrt(d_model),
        size=(d_model, head_dim)
    )
    for _ in range(num_heads)
]

# After concatenating both heads, project back to d_model.
W_O = rng.normal(
    0.0, 1 / np.sqrt(d_model),
    size=(num_heads * head_dim, d_model)
)


# ------------------------------------------------------------
# PART 6: Calculate multi-head causal self-attention
# ------------------------------------------------------------

def multi_head_attention(x):
    """
    x shape: (sequence_length, d_model)

    Returns:
        output: (sequence_length, d_model)
        all_weights: one attention matrix per head
    """

    head_outputs = []
    all_weights = []

    for head_index in range(num_heads):

        # Calculate this head's Q, K, and V from the input.
        # No Q/K/V values are manually entered.
        Q = x @ W_Q[head_index]
        K = x @ W_K[head_index]
        V = x @ W_V[head_index]

        # Compare every query with every key.
        scores = (Q @ K.T) / np.sqrt(head_dim)

        # Prevent each token from attending to future tokens.
        future_mask = np.triu(
            np.ones(scores.shape, dtype=bool),
            k=1
        )

        # Negative infinity becomes zero probability in softmax.
        scores[future_mask] = -np.inf

        # Stable row-wise softmax.
        scores = scores - np.max(
            scores, axis=1, keepdims=True
        )

        exponential_scores = np.exp(scores)

        weights = exponential_scores / np.sum(
            exponential_scores, axis=1, keepdims=True
        )

        # Combine the value vectors according to attention.
        head_output = weights @ V

        head_outputs.append(head_output)
        all_weights.append(weights)

        # Inspect the first head's shapes for learning.
        if head_index == 0:
            print("\nFirst head Q shape:", Q.shape)
            print("First head K shape:", K.shape)
            print("First head V shape:", V.shape)
            print("First head attention weights:")
            print(weights)

    # Join the output features from all heads.
    concatenated = np.concatenate(head_outputs, axis=1)

    # Mix the concatenated features and restore d_model.
    output = concatenated @ W_O

    return output, all_weights


# ------------------------------------------------------------
# PART 7: First sublayer — attention + residual connection
# ------------------------------------------------------------

# Pre-LN: normalize the input before attention.
normalized_X = layer_norm(X, gamma1, beta1)

# Calculate attention using automatically computed Q/K/V.
attention_output, attention_weights = multi_head_attention(
    normalized_X
)

print("\nAttention output shape:", attention_output.shape)

# Residual connection: retain the original representation.
# Both matrices have shape (sequence_length, d_model).
after_attention = X + attention_output

print("After first residual:", after_attention.shape)


# ------------------------------------------------------------
# PART 8: Initialize the feed-forward network
# ------------------------------------------------------------

# First linear layer:
# d_model -> hidden_dim
W1 = rng.normal(
    0.0, 1 / np.sqrt(d_model),
    size=(d_model, hidden_dim)
)
b1 = np.zeros(hidden_dim)

# Second linear layer:
# hidden_dim -> d_model
W2 = rng.normal(
    0.0, 1 / np.sqrt(hidden_dim),
    size=(hidden_dim, d_model)
)
b2 = np.zeros(d_model)


# ------------------------------------------------------------
# PART 9: Second sublayer — feed-forward network
# ------------------------------------------------------------

# Normalize before applying the feed-forward network.
normalized_H = layer_norm(
    after_attention, gamma2, beta2
)

# First linear transformation.
hidden = normalized_H @ W1 + b1

# ReLU introduces a nonlinear transformation.
hidden = np.maximum(hidden, 0)

# Second linear transformation restores d_model features.
feed_forward_output = hidden @ W2 + b2

print("\nFeed-forward hidden shape:", hidden.shape)
print("Feed-forward output shape:", feed_forward_output.shape)


# ------------------------------------------------------------
# PART 10: Second residual connection
# ------------------------------------------------------------

# Preserve the input to the feed-forward sublayer as well.
block_output = after_attention + feed_forward_output

print("\nFinal Transformer block output:")
print(block_output)
print("Final output shape:", block_output.shape)


# ------------------------------------------------------------
# PART 11: Verify the architecture
# ------------------------------------------------------------

assert X.shape == (sequence_length, d_model)

assert attention_output.shape == (
    sequence_length, d_model
)

assert after_attention.shape == (
    sequence_length, d_model
)

assert feed_forward_output.shape == (
    sequence_length, d_model
)

assert block_output.shape == (
    sequence_length, d_model
)

# Each attention head's rows should sum to one.
for weights in attention_weights:
    assert np.allclose(
        weights.sum(axis=1),
        np.ones(sequence_length)
    )

    # Future positions must receive zero weight.
    assert np.allclose(
        np.triu(weights, k=1),
        0.0
    )

print("\nAll Transformer block checks passed!")

