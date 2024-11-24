import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import torch
import torch.nn.functional as F

def plot_tokens_with_colored_background(tokens, token_scores, path_end=None, save_path=None):
    """
    Plots tokens with individual background colors and small gaps between tokens.

    Parameters:
    - tokens (list of str): List of tokens to plot.
    - token_scores (torch.Tensor or numpy.ndarray): Scores associated with each token.
    - save_path (str): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Convert token_scores to numpy array if it's a torch.Tensor
    if isinstance(token_scores, torch.Tensor):
        token_scores = token_scores.detach().cpu().numpy()

    # Normalize token scores between 0 and 1
    eps = 1e-8
    min_score = token_scores.min()
    max_score = token_scores.max()
    token_scores = (token_scores - min_score) / (max_score - min_score + eps)

    # Create a color map based on the token scores
    cmap = plt.get_cmap('Reds')
    colors = [cmap(score) for score in token_scores]

    # Plot tokens with colored backgrounds
    ax = plt.gca()
    ax.axis('off')  # Hide the axes

    x = 0.01  # Starting x position
    y = 1.0   # Starting y position
    line_height = 0.1  # Adjust for line spacing
    space_width = 0.005  # Small gap between tokens

    for token, color in zip(tokens, colors):
        # Plot the token with background color
        text = ax.text(
            x, y, token, fontsize=12, color='black', ha='left', va='top',
            bbox=dict(facecolor=color, edgecolor='none', boxstyle='square,pad=0.1')
        )
        # Update x position based on text width
        renderer = plt.gcf().canvas.get_renderer()
        text_width = text.get_window_extent(renderer=renderer).width
        # Convert pixel width to data coordinates
        inv = ax.transData.inverted()
        text_width_data = inv.transform((text_width, 0))[0] - inv.transform((0, 0))[0]
        x += text_width_data + space_width

        # Wrap text if x exceeds plot width
        if x > 0.99:
            x = 0.01
            y -= line_height

    # Save or display the plot
    if save_path:
        if path_end is not None:
            save_path = f"{save_path}_{path_end}.png"
        else:
            save_path = f"{save_path}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_sentences_with_colored_background(tokens, token_scores, sentence_indices, path_end=None, save_path=None):
    """
    Plots tokens grouped by sentences, with separations only between sentences.
    
    Parameters:
    - tokens (list of str): List of tokens to plot.
    - token_scores (torch.Tensor or numpy.ndarray): Scores associated with each token.
    - sentence_indices (list or numpy.ndarray): Sentence indices corresponding to each token.
    - save_path (str): Path to save the plot. If None, the plot will be displayed.
    """
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Convert token_scores and sentence_indices to numpy arrays if they are torch.Tensors
    if isinstance(token_scores, torch.Tensor):
        token_scores = token_scores.detach().cpu().numpy()
    if isinstance(sentence_indices, torch.Tensor):
        sentence_indices = sentence_indices.detach().cpu().numpy()

    # Normalize token scores between 0 and 1
    eps = 1e-8
    min_score = token_scores.min()
    max_score = token_scores.max()
    token_scores = (token_scores - min_score) / (max_score - min_score + eps)

    # Create a color map based on the token scores
    cmap = plt.get_cmap('Reds')
    colors = [cmap(score) for score in token_scores]

    # Group tokens and colors by sentences
    sentences = {}
    for token, color, idx in zip(tokens, colors, sentence_indices):
        if idx not in sentences:
            sentences[idx] = {'tokens': [], 'colors': []}
        sentences[idx]['tokens'].append(token)
        sentences[idx]['colors'].append(color)

    # Plot sentences with separations only between sentences
    ax = plt.gca()
    ax.axis('off')  # Hide the axes

    x = 0.01  # Starting x position
    y = 1.0   # Starting y position
    line_height = 0.1  # Adjust for line spacing
    sentence_gap = 0.02  # Gap between sentences

    for idx in sorted(sentences.keys()):
        tokens_line = sentences[idx]['tokens']
        colors_line = sentences[idx]['colors']

        for token, color in zip(tokens_line, colors_line):
            # Plot the token with background color
            text = ax.text(
                x, y, token, fontsize=12, color='black', ha='left', va='top',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='square,pad=0.0')
            )
            # Update x position based on text width
            renderer = plt.gcf().canvas.get_renderer()
            text_width = text.get_window_extent(renderer=renderer).width
            # Convert pixel width to data coordinates
            inv = ax.transData.inverted()
            text_width_data = inv.transform((text_width, 0))[0] - inv.transform((0, 0))[0]
            x += text_width_data  # No gap between tokens

            # Wrap text if x exceeds plot width
            if x > 0.99:
                x = 0.01
                y -= line_height

        # Add a gap between sentences
        x += sentence_gap
        # Wrap text if x exceeds plot width after adding sentence gap
        if x > 0.99:
            x = 0.01
            y -= line_height

    # Save or display the plot
    if save_path:
        if path_end is not None:
            save_path = f"{save_path}_{path_end}.png"
        else:
            save_path = f"{save_path}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
