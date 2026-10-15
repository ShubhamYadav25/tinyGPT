
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
