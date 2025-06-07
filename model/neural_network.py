import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

from utility_classes import ResidualBlock
from data_conversion import ProblemEncoder, OutputDecoder

class ARCNeuralNetwork(nn.Module):
    """
    A conceptual neural network for processing 30x30 images with 10 colors,
    designed to extract features or generate programs for a symbolic AI component.
    Now enhanced with Residual Blocks, GELU activation, and awareness of true input dimensions.
    Also includes a method to embed individual objects.
    """
    def __init__(self, num_colors=10, grid_size=30, latent_dim=256, program_vocab_size=100, num_predicates=50):
        """
        Initializes the neural network architecture.

        Args:
            num_colors (int): The number of possible colors in the input grid (e.g., 10).
            grid_size (int): The dimension of the square input grid (e.g., 30 for 30x30).
            latent_dim (int): The dimension of the aggregated latent representation.
            program_vocab_size (int): The size of the vocabulary for the Domain-Specific Language (DSL)
                                      if the network is generating programs.
            num_predicates (int): The number of distinct symbolic predicates the network can output.
        """
        super(ARCNeuralNetwork, self).__init__()

        self.num_colors = num_colors
        self.grid_size = grid_size
        self.latent_dim = latent_dim # Storing latent_dim for object embedding size

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(num_colors, 64, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)

        # Use Residual Blocks for feature extraction
        # Block 1: 64 -> 128 channels, no downsampling
        self.block1 = ResidualBlock(64, 128, stride=1)
        # Block 2: 128 -> 256 channels, downsamples by 2 (due to stride=2)
        self.block2 = ResidualBlock(128, 256, stride=2) # Halves spatial dimensions

        # Calculate the size of the flattened feature map after convolutions
        # Output of block2 is 256 channels, spatial dim is grid_size // 2
        self.flattened_feature_map_size = 256 * (grid_size // 2) * (grid_size // 2)

        # Fully connected layers to aggregate grid features into a latent representation
        # Input to fc_agg1 now includes the flattened feature map PLUS one-hot encoded dimensions (grid_size * 2)
        self.fc_agg1 = nn.Linear(self.flattened_feature_map_size + (grid_size * 2), latent_dim * 2)
        self.fc_agg2 = nn.Linear(latent_dim * 2, latent_dim)

        # --- New: Object Embedding Branch ---
        # MLP to embed individual object properties into a fixed-size vector (latent_dim)
        # Input features for an object:
        # 1. Color (one-hot encoded: num_colors)
        # 2. Bounding Box (min_r, min_c, max_r, max_c: 4 scalars)
        # 3. Size (1 scalar)
        object_feature_dim = num_colors + 4 + 1 # num_colors for one-hot, 4 for bbox, 1 for size
        self.object_embedding_mlp = nn.Sequential(
            nn.Linear(object_feature_dim, latent_dim), # Each object is embedded to latent_dim
            nn.GELU()
        )

        # Output heads (conceptual, for program generation or predicate prediction)
        self.program_output_linear = nn.Linear(latent_dim, program_vocab_size)
        self.predicate_output_linear = nn.Linear(latent_dim, num_predicates)

    def _encode(self, x, input_height, input_width):
        """
        Internal helper for encoding an image (grid-level features).
        x: (batch_size, num_colors, grid_size, grid_size)
        input_height: (batch_size,) - true height of the grid
        input_width: (batch_size,) - true width of the grid
        """
        batch_size = x.size(0)

        x = F.gelu(self.initial_bn(self.initial_conv(x)))
        x = self.block1(x)
        x = self.block2(x)

        # Flatten the feature map
        x = x.view(batch_size, -1)

        # One-hot encode height and width and concatenate to the features
        height_one_hot = F.one_hot(input_height - 1, num_classes=self.grid_size).float()
        width_one_hot = F.one_hot(input_width - 1, num_classes=self.grid_size).float()

        # Concatenate spatial features with one-hot encoded dimensions
        x_with_dims = torch.cat([x, height_one_hot, width_one_hot], dim=1)

        x = F.gelu(self.fc_agg1(x_with_dims))
        latent_representation = F.gelu(self.fc_agg2(x))
        return latent_representation

    def _encode_object(self, obj_data, grid_h, grid_w):
        """
        Encodes a single object dictionary into a fixed-size tensor.
        obj_data: {'color': int, 'pixels': list, 'bbox': tuple, 'size': int}
        grid_h, grid_w: The true height and width of the grid the object came from (for normalization).
        """
        # Convert color to one-hot
        color_one_hot = F.one_hot(torch.tensor(obj_data['color'], dtype=torch.long, device=self.object_embedding_mlp[0].weight.device), num_classes=self.num_colors).float()

        # Normalize bounding box coordinates and size by grid dimensions
        min_r, min_c, max_r, max_c = obj_data['bbox']
        bbox_norm = torch.tensor([
            min_r / (grid_h - 1e-5), min_c / (grid_w - 1e-5), # Use current grid_h/w for normalization
            max_r / (grid_h - 1e-5), max_c / (grid_w - 1e-5)
        ], dtype=torch.float, device=self.object_embedding_mlp[0].weight.device)
        size_norm = torch.tensor([obj_data['size'] / (grid_h * grid_w + 1e-5)], dtype=torch.float, device=self.object_embedding_mlp[0].weight.device) # Normalize by total grid pixels

        # Concatenate all features and pass through MLP
        object_features = torch.cat([color_one_hot, bbox_norm, size_norm])
        
        return self.object_embedding_mlp(object_features)

    def forward(self, x, input_height, input_width):
        """
        Defines the forward pass of the neural network for its original conceptual purpose.
        Returns program and predicate logits.
        """
        latent_representation = self._encode(x, input_height, input_width)
        program_logits = self.program_output_linear(latent_representation)
        predicate_logits = self.predicate_output_linear(latent_representation)
        return program_logits, predicate_logits

    def get_latent_representation(self, x, input_height, input_width):
        """
        Provides the latent representation of an input image,
        useful when this network acts as an encoder in a larger system.
        """
        return self._encode(x, input_height, input_width)


class FullARCModel(nn.Module):
    """
    Combines the ImageEncoder, ProblemEncoder, and OutputDecoder
    to solve the full ARC-like problem.
    Now correctly passes input dimensions and object data to the problem encoder.
    """
    def __init__(self, num_colors=10, grid_size=30, latent_dim=256, num_train_pairs=2):
        super(FullARCModel, self).__init__()
        self.image_encoder = ARCNeuralNetwork(num_colors, grid_size, latent_dim)
        self.problem_encoder = ProblemEncoder(latent_dim, num_train_pairs)
        self.output_decoder = OutputDecoder(latent_dim, num_colors, grid_size)
        self.num_train_pairs = num_train_pairs

    def forward(self, problem_batch):
        """
        Processes a batch of ARC problems.

        Args:
            problem_batch (dict): A dictionary containing batched tensors and lists for:
                'train_inputs': List of `num_train_pairs` tensors, each (B, num_colors, G, G)
                'train_outputs': List of `num_train_pairs` tensors, each (B, num_colors, G, G)
                'train_input_dims': List of `num_train_pairs` tensors, each (B, 2) for (height, width)
                'train_output_dims': List of `num_train_pairs` tensors, each (B, 2) for (height, width)
                'train_input_objects': List of `num_train_pairs` lists, each list contains (B) lists of object dicts
                'train_output_objects': List of `num_train_pairs` lists, each list contains (B) lists of object dicts
                'test_input': Tensor (B, num_colors, G, G)
                'test_input_dims': Tensor (B, 2) for (height, width)
                'test_input_objects': List of (B) lists of object dicts

        Returns:
            tuple:
                - predicted_output_logits (Tensor): Predicted output grid (logits for each color at each pixel).
                - predicted_height_logits (Tensor): Logits for predicted output height.
                - predicted_width_logits (Tensor): Logits for predicted output width.
        """
        batch_size = problem_batch['test_input'].size(0)

        # Encode all training input grids, passing their true dimensions
        encoded_train_inputs = []
        for i in range(self.num_train_pairs):
            encoded_train_inputs.append(self.image_encoder.get_latent_representation(
                problem_batch['train_inputs'][i],
                problem_batch['train_input_dims'][i][:, 0], # True heights for this train input pair
                problem_batch['train_input_dims'][i][:, 1]  # True widths for this train input pair
            ))

        # Encode all training output grids, passing their true dimensions
        encoded_train_outputs = []
        for i in range(self.num_train_pairs):
            encoded_train_outputs.append(self.image_encoder.get_latent_representation(
                problem_batch['train_outputs'][i],
                problem_batch['train_output_dims'][i][:, 0], # True heights for this train output pair
                problem_batch['train_output_dims'][i][:, 1]  # True widths for this train output pair
            ))

        # Encode the test input grid, passing its true dimensions
        encoded_test_input = self.image_encoder.get_latent_representation(
            problem_batch['test_input'],
            problem_batch['test_input_dims'][:, 0], # True height for test input
            problem_batch['test_input_dims'][:, 1]  # True width for test input
        )

        # Infer the rule embedding using the ProblemEncoder, passing both grid and object data
        # Pass a dictionary of dimension tensors to ProblemEncoder for convenience
        problem_batch_dims = {
            'train_input_dims': problem_batch['train_input_dims'],
            'train_output_dims': problem_batch['train_output_dims'],
            'test_input_dims': problem_batch['test_input_dims']
        }
        
        rule_embedding = self.problem_encoder(
            encoded_train_inputs,
            encoded_train_outputs,
            encoded_test_input,
            problem_batch['train_input_objects'], # Pass object data
            problem_batch['train_output_objects'], # Pass object data
            problem_batch['test_input_objects'], # Pass object data
            self.image_encoder, # Pass reference to image_encoder for object embedding
            problem_batch_dims # Pass the dictionary of dimension tensors
        )

        # Generate the final output and predict its size using the OutputDecoder
        # OutputDecoder now receives a rule_embedding that is implicitly richer from object-level processing.
        predicted_output_logits, predicted_height_logits, predicted_width_logits = \
            self.output_decoder(rule_embedding, encoded_test_input)

        return predicted_output_logits, predicted_height_logits, predicted_width_logits