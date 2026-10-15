
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
