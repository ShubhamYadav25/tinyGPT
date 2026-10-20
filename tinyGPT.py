"""
Training text
"hello world!\nhello there!..."

Tokenization
Convert characters to integer IDs

Token + positional embeddings
Represent identity and position

Transformer block × 2
Causal multi-head attention + FFN + residuals + LayerNorm

Language-model head
Convert final representations into vocabulary logits

Cross-entropy loss + AdamW
Learn from the correct next characters

Autoregressive generation
Predict a character, append it, and repeat


What does the model learn during training?
1. Forward pass
Calculate logits for the next characters.

2. Cross-entropy loss
Measure how much probability the correct targets receive.

3. Backpropagation
Automatically calculate gradients through every trainable layer.

4. AdamW update
Update parameters to improve future predictions.
"""


# ============================================================
# STEP 14: BUILD AND TRAIN A TINY GPT FROM SCRATCH
#
# Architecture:
#   Character IDs
#       -> Token + positional embeddings
#       -> Transformer block 1
#       -> Transformer block 2
#       -> Final layer normalization
#       -> Language-model head
#       -> Next-character logits
#
# Training:
#   Next-character targets -> Cross-entropy loss
#   -> Automatic backpropagation -> AdamW updates
#
# Generation:
#   Prompt -> Predict next character -> Append -> Repeat
#
# Important:
#   Q, K, V are calculated from trainable linear projections.
#   No Q/K/V vectors are manually entered.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# PART 1: Reproducibility and device selection
# ------------------------------------------------------------

# Fix the random seed to make initial parameters and sampling
# reproducible on the same device and software setup.
torch.manual_seed(42)

# Use a GPU when available; otherwise, use the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)



# ------------------------------------------------------------
# PART 2: Create a tiny Indian-related Q&A dataset
# ------------------------------------------------------------

# Each example teaches the model a text pattern.
# We include questions and answers so it can learn how
# such exchanges are written.
training_examples = [
    # Indian geography
    "Question: What is the capital of India?\nAnswer: New Delhi.",
    "Question: What is the capital of Maharashtra?\nAnswer: Mumbai.",
    "Question: What is the capital of Rajasthan?\nAnswer: Jaipur.",
    "Question: What is the capital of Tamil Nadu?\nAnswer: Chennai.",
    "Question: What is the capital of Karnataka?\nAnswer: Bengaluru.",
    "Question: What is the capital of West Bengal?\nAnswer: Kolkata.",
    "Question: What is the capital of Uttar Pradesh?\nAnswer: Lucknow.",

    # Indian history and national observances
    "Question: When is Independence Day in India?\nAnswer: 15 August.",
    "Question: When is Republic Day in India?\nAnswer: 26 January.",

    # Indian landmarks and currency
    "Question: Where is the Taj Mahal?\nAnswer: The Taj Mahal is in Agra.",
    "Question: What is the currency of India?\nAnswer: The Indian currency is the rupee.",

    # General facts
    "Fact: India is a country in South Asia.",
    "Fact: The Ganga is a major river in India.",
    "Fact: The Himalayas lie in northern India.",
    "Fact: India has many languages and cultures.",
    "Fact: Cricket is a popular sport in India.",
    "Fact: The Indian national flag has saffron, white, and green stripes and a blue Ashoka Chakra.",

    # A few examples from our earlier lessons
    "Fact: Attention uses queries, keys, and values.",
    "Fact: A tiny language model learns patterns from text."
]

# Join the examples, then repeat the small dataset.
# Repetition gives our tiny model many opportunities to see
# the same patterns. It does not add new factual knowledge.
text = ("\n".join(training_examples) + "\n") * 80

# Build the character vocabulary from the complete text.
characters = sorted(set(text))
vocab_size = len(characters)

char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

print("Training text length:", len(text))
print("Number of examples:", len(training_examples))
print("Vocabulary size:", vocab_size)
print("Vocabulary:", repr("".join(characters)))


# Convert readable text into integer IDs.
def encode(text_to_encode):
    """Convert text into a list of character IDs."""

    unknown = set(text_to_encode) - set(char_to_id)

    if unknown:
        raise ValueError(
            f"Characters not in vocabulary: {unknown!r}"
        )

    return [
        char_to_id[character]
        for character in text_to_encode
    ]


# Convert character IDs back into readable text.
def decode(token_ids):
    """Convert character IDs into a string."""

    return "".join(
        id_to_char[int(token_id)]
        for token_id in token_ids
    )

print("\nDataset length:", len(text))
print("Vocabulary size:", vocab_size)
print("Vocabulary:", repr("".join(characters)))


# ------------------------------------------------------------
# PART 3: Convert the dataset into a PyTorch tensor
# ------------------------------------------------------------

# The complete dataset becomes one long sequence of integers.
all_token_ids = torch.tensor(
    encode(text),
    dtype=torch.long
)

# Keep the last 10% for a simple validation-loss estimate.
# The remaining 90% is used for training.
split_index = int(0.9 * len(all_token_ids))

train_data = all_token_ids[:split_index]
val_data = all_token_ids[split_index:]

print("\nTraining token count:", len(train_data))
print("Validation token count:", len(val_data))


# ------------------------------------------------------------
# PART 4: Configure the tiny GPT
# ------------------------------------------------------------

# block_size:
# Maximum number of characters the model can process at once.
block_size = 24

# d_model:
# Number of features representing each character.
d_model = 32

# num_heads:
# Number of independent attention heads in each block.
num_heads = 4

# num_layers:
# Number of Transformer blocks stacked together.
num_layers = 2

# Training settings.
batch_size = 32
max_iters = 1000
eval_interval = 200
eval_iters = 20
learning_rate = 3e-4

# Each head receives an equal portion of model dimensions.
assert d_model % num_heads == 0

print("\nModel configuration:")
print("Context length:", block_size)
print("Model dimensions:", d_model)
print("Attention heads:", num_heads)
print("Transformer blocks:", num_layers)


# ------------------------------------------------------------
# PART 5: Create training batches
# ------------------------------------------------------------

def get_batch(source_data):
    """
    Produce a batch of input sequences and next-character targets.

    X contains the input characters.
    Y contains the same sequences shifted one character right.

    Example:
        Input:  hello wor
        Target: ello worl

    The model learns to predict the next character at
    every position in the sequence.
    """

    # Choose random starting positions.
    starts = torch.randint(
        low=0,
        high=len(source_data) - block_size,
        size=(batch_size,)
    )

    # Build input sequences.
    X = torch.stack([
        source_data[int(start):int(start) + block_size]
        for start in starts
    ])

    # Shift every input sequence by one character to create Y.
    Y = torch.stack([
        source_data[
            int(start) + 1:int(start) + block_size + 1
        ]
        for start in starts
    ])

    # Move the batch to the selected CPU or GPU.
    return X.to(device), Y.to(device)


# ------------------------------------------------------------
# PART 6: Implement causal self-attention
# ------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    One multi-head causal self-attention sublayer.

    The input contains a batch of token representations.
    Trainable projections calculate Q, K, and V automatically.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Trainable projection matrices, implemented as Linear layers.
        # PyTorch initializes their parameters automatically.
        #
        # Each projection transforms d_model features into d_model
        # features. Those features are split across attention heads.
        #
        # These are NOT hardcoded Q, K, or V values.
        self.q_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        self.k_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        self.v_proj = nn.Linear(
            d_model, d_model, bias=False
        )

        # Combine all head outputs and project back to d_model.
        self.out_proj = nn.Linear(
            d_model, d_model
        )

    def forward(self, x):
        """
        Input shape: (batch_size, sequence_length, d_model)
        Output shape: (batch_size, sequence_length, d_model)
        """

        B, T, C = x.shape

        # ----------------------------------------------------
        # A: Calculate Q, K, and V from the actual input.
        # ----------------------------------------------------

        # Each operation uses a separate trainable projection.
        # Q, K, V initially have shape (B, T, d_model).
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ----------------------------------------------------
        # B: Split the feature dimension into attention heads.
        # ----------------------------------------------------

        # Reshape:
        # (B, T, d_model) -> (B, T, num_heads, head_dim)
        #
        # Transpose:
        # (B, T, num_heads, head_dim)
        # -> (B, num_heads, T, head_dim)
        #
        # Each head now operates on its own feature subspace.
        Q = Q.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # C: Compare queries with keys.
        # ----------------------------------------------------

        # Q shape:   (B, heads, T, head_dim)
        # K transpose:
        #            (B, heads, head_dim, T)
        #
        # Scores shape: (B, heads, T, T)
        #
        # The last two dimensions contain token-to-token scores.
        scores = (
            Q @ K.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # ----------------------------------------------------
        # D: Apply the causal mask.
        # ----------------------------------------------------

        # True above the diagonal means "future position".
        # Every token must be prevented from reading future tokens.
        future_mask = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        # Masked scores become negative infinity.
        # Their softmax probabilities will be zero.
        scores = scores.masked_fill(
            future_mask,
            float("-inf")
        )

        # ----------------------------------------------------
        # E: Turn attention scores into attention weights.
        # ----------------------------------------------------

        # Softmax is applied across the key positions.
        # Each query's allowed attention weights sum to 1.
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # F: Combine value vectors.
        # ----------------------------------------------------

        # Each query gets a weighted combination of V.
        # Shape remains (B, heads, T, head_dim).
        out = attention_weights @ V

        # Rejoin all head features.
        # (B, heads, T, head_dim)
        # -> (B, T, heads, head_dim)
        out = out.transpose(1, 2).contiguous()

        # Merge heads into the original model dimension.
        out = out.view(B, T, C)

        # Learn how to mix information from all heads.
        out = self.out_proj(out)

        return out


# ------------------------------------------------------------
# PART 7: Build one Transformer block
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A pre-layer-normalization Transformer block.

    Components:
        LayerNorm -> Multi-head attention -> Residual
        LayerNorm -> Feed-forward network -> Residual
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        # Normalize features before attention.
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-head causal self-attention.
        self.attention = CausalSelfAttention(
            d_model, num_heads
        )

        # Normalize features before the feed-forward network.
        self.ln2 = nn.LayerNorm(d_model)

        # The feed-forward network operates independently
        # on the feature vector at each token position.
        #
        # Expand d_model -> 4*d_model, apply GELU,
        # then project back to d_model.
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):

        # Residual connection around attention.
        # Preserve the input while adding contextual information.
        x = x + self.attention(
            self.ln1(x)
        )

        # Residual connection around the feed-forward network.
        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


# ------------------------------------------------------------
# PART 8: Assemble the GPT-style language model
# ------------------------------------------------------------

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        block_size
    ):
        super().__init__()

        self.block_size = block_size

        # Learn a vector for each vocabulary character.
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        # Learn a vector for each position in the context window.
        self.position_embedding = nn.Embedding(
            block_size, d_model
        )

        # Stack several Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # Normalize the final token representations.
        self.final_norm = nn.LayerNorm(d_model)

        # Map each token's final representation to vocabulary logits.
        # One logit per possible next character.
        self.lm_head = nn.Linear(
            d_model, vocab_size
        )

    def forward(self, token_ids):
        """
        token_ids shape:
            (batch_size, sequence_length)

        Returns logits with shape:
            (batch_size, sequence_length, vocab_size)
        """

        B, T = token_ids.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds context limit "
                f"{self.block_size}."
            )

        # Look up token embeddings.
        token_vectors = self.token_embedding(token_ids)

        # Create position IDs 0, 1, ..., T-1.
        positions = torch.arange(
            T, device=token_ids.device
        )

        # Look up positional embeddings.
        position_vectors = self.position_embedding(positions)

        # Combine token identity and positional information.
        # The same positional matrix broadcasts across the batch.
        x = token_vectors + position_vectors.unsqueeze(0)

        # Process the sequence through every Transformer block.
        for block in self.blocks:
            x = block(x)

        # Final normalization.
        x = self.final_norm(x)

        # Produce one score for each possible next character.
        logits = self.lm_head(x)

        return logits


# ------------------------------------------------------------
# PART 9: Initialize the model
# ------------------------------------------------------------

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    block_size=block_size
).to(device)

# Count all trainable parameters.
parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTotal trainable parameters:", parameter_count)


# ------------------------------------------------------------
# PART 10: Estimate training and validation loss
# ------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """
    Estimate average cross-entropy loss on training and
    validation batches without calculating gradients.
    """

    model.eval()

    results = {}

    for split_name, split_data in [
        ("train", train_data),
        ("validation", val_data)
    ]:

        losses = []

        for _ in range(eval_iters):

            X, Y = get_batch(split_data)

            # Predict next-character logits at every position.
            logits = model(X)

            # Cross-entropy expects class scores and target IDs.
            # Flatten batch and sequence dimensions together.
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                Y.reshape(-1)
            )

            losses.append(loss.item())

        results[split_name] = sum(losses) / len(losses)

    model.train()

    return results


# ------------------------------------------------------------
# PART 11: Set up the optimizer
# ------------------------------------------------------------

# AdamW updates all trainable parameters, including:
# embeddings, Q/K/V projections, output projections,
# feed-forward layers, normalization parameters, and lm_head.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ------------------------------------------------------------
# PART 12: Train the model
# ------------------------------------------------------------

print("\nStarting training...")

for step in range(max_iters):

    # Periodically examine progress on both data splits.
    if step % eval_interval == 0:
        losses = estimate_loss()

        print(
            f"Step {step:4} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['validation']:.4f}"
        )

    # Get a random mini-batch.
    X, Y = get_batch(train_data)

    # Forward pass: produce next-character predictions.
    logits = model(X)

    # Calculate cross-entropy against the shifted targets.
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )

    # Clear gradients from the previous training iteration.
    optimizer.zero_grad(set_to_none=True)

    # PyTorch automatically calculates gradients through the
    # entire network, including attention and embeddings.
    loss.backward()

    # Keep the overall gradient size under control.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Apply the parameter updates.
    optimizer.step()


# Display final loss estimates.
final_losses = estimate_loss()

print("\nTraining finished!")
print(f"Final train loss: {final_losses['train']:.4f}")
print(f"Final validation loss: {final_losses['validation']:.4f}")



# ------------------------------------------------------------
# PART 13: Stream generated text as it is produced
# ------------------------------------------------------------

@torch.no_grad()
def generate_stream(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.3,
    stop_at_newline=True
):
    """
    Generate characters one at a time and print immediately.

    This resembles streamed chat output:
    the user sees each character as soon as it is generated.

    Since our tokenizer operates at character level, we stream
    characters rather than word or subword tokens.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    # Remember whether the model was in training mode.
    was_training = model.training

    # Disable dropout and other training-only behavior.
    model.eval()

    # Encode the starting question or prompt.
    generated_ids = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    # Show the prompt immediately, without ending the line.
    print(prompt, end="", flush=True)

    try:
        for _ in range(max_new_tokens):

            # The model can only process block_size characters.
            # Keep the most recent context characters.
            context_ids = generated_ids[:, -block_size:]

            # Predict next-character logits for the context.
            logits = model(context_ids)

            # Only the final position predicts the next character.
            next_logits = logits[:, -1, :]

            # Temperature controls the randomness of sampling.
            next_logits = next_logits / temperature

            # Convert scores into a probability distribution.
            probabilities = F.softmax(
                next_logits,
                dim=-1
            )

            # Sample the next character from those probabilities.
            next_id = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Convert the predicted ID back into a character.
            next_character = id_to_char[int(next_id.item())]

            # Add the new character to the sequence.
            # It becomes part of the next prediction's context.
            generated_ids = torch.cat(
                [generated_ids, next_id],
                dim=1
            )

            # Print immediately instead of buffering the response.
            print(next_character, end="", flush=True)

            # For Q&A prompts, stop when an answer-ending newline
            # is generated. Otherwise, continue up to the limit.
            if stop_at_newline and next_character == "\n":
                break

    finally:
        # Restore the model's original training/evaluation mode.
        model.train(was_training)

    # End the streamed response cleanly.
    print()


# ------------------------------------------------------------
# PART 14: Try streaming some Indian-related answers
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("STREAMING GENERATION")
print("=" * 55)

# The prompts match prefixes present in our training examples.
prompts = [
    "Question: What is the capital of India?\nAnswer:",
    "Question: Where is the Taj Mahal?\nAnswer:",
    "Question: What is the currency of India?\nAnswer:"
]

for prompt in prompts:

    print("\n--- New response ---")

    generate_stream(
        model=model,
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.3,
        stop_at_newline=True
    )

