import re
from transformers import AutoTokenizer  # For initializing or using the tokenizer

def split_paragraph_to_sentences_with_offsets(paragraph):
    # Split by punctuation and keep the spaces intact in split sentences
    sentence_boundaries = re.finditer(r'(?<=[.,!?])\s', paragraph)
    
    # Identify where each sentence begins and ends based on split points
    start = 0
    sentences = []
    offsets = []
    for match in sentence_boundaries:
        end = match.end() - 1
        # sentences.append(paragraph[start:end].strip())
        sentences.append(paragraph[start:end])
        offsets.append((start, end))
        start = end

    # Add any remaining text as the final sentence
    if start < len(paragraph):
        # sentences.append(paragraph[start:].strip())
        sentences.append(paragraph[start:])
        offsets.append((start, len(paragraph)))

    return sentences, offsets

def map_tokens_to_sentence_indices(tokenizer, input_text):
    # Tokenize the entire input text to get full token list
    full_tokenization = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    full_tokens = full_tokenization["input_ids"][0].tolist()
    token_offsets = tokenizer.encode_plus(input_text, return_offsets_mapping=True)["offset_mapping"]

    # Split the text into sentences with offsets
    sentences, sentence_offsets = split_paragraph_to_sentences_with_offsets(input_text)

    # Map tokens to sentence indices based on offsets
    sentence_indices = []
    for token_offset in token_offsets:
        # Identify the sentence index based on token position
        sentence_idx = next(
            (idx for idx, (start, end) in enumerate(sentence_offsets) if start <= token_offset[0] < end),
            len(sentences) - 1
        )
        sentence_indices.append(sentence_idx)

    return full_tokenization, sentence_indices
