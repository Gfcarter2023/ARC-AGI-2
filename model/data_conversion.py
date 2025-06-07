import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
from utility_classes import ConditionalBatchNorm2d


class ProblemEncoder(nn.Module):
    """
    Conceptual module to infer a 'rule embedding' from multiple
    input-output pairs and the test input's representation.
    Now uses a Transformer Encoder for relational reasoning and positional embeddings,
    and also incorporates object-level reasoning.
    """
    def __init__(self, latent_dim=256, num_train_pairs=2, num_heads=4, max_objects_per_grid=50):
        super(ProblemEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_train_pairs = num_train_pairs
        self.num_heads = num_heads
        self.max_objects_per_grid = max_objects_per_grid # Max objects to process per grid

        # The sequence length for the main Transformer:
        # num_train_pairs (for transformation embeddings) + 1 (for global test input embedding)
        self.sequence_len = num_train_pairs + 1

        # Positional embeddings for each slot in the main sequence
        self.positional_embeddings = nn.Parameter(torch.randn(self.sequence_len, latent_dim))

        # --- Object-level Transformer ---
        # To reason about objects within a single grid (input or output).
        # We'll input MAX_OBJECTS_PER_GRID embeddings + a CLS token.
        self.object_seq_len = max_objects_per_grid + 1 # +1 for a CLS token
        self.object_cls_token = nn.Parameter(torch.randn(1, latent_dim)) # CLS token for object aggregation
        self.object_positional_embeddings = nn.Parameter(torch.randn(self.object_seq_len, latent_dim))

        object_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2, # Smaller feedforward for object-level
            dropout=0.1,
            batch_first=True
        )
        self.object_transformer_encoder = nn.TransformerEncoder(object_encoder_layer, num_layers=1)

        # MLP to combine grid-level and object-level embeddings for a unified representation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), # latent_dim (from grid) + latent_dim (from objects)
            nn.GELU()
        )

        # Main Transformer Encoder (processes sequence of fused embeddings)
        main_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.main_transformer_encoder = nn.TransformerEncoder(main_encoder_layer, num_layers=1)

    def forward(self, encoded_train_inputs, encoded_train_outputs, encoded_test_input,
                train_input_objects_batch, train_output_objects_batch, test_input_objects_batch,
                image_encoder_ref, problem_batch_dims): # Added problem_batch_dims for true H/W
        """
        Args:
            encoded_train_inputs (list of Tensors): Latent representations of training inputs (grid-level).
            encoded_train_outputs (list of Tensors): Latent representations of training outputs (grid-level).
            encoded_test_input (Tensor): Latent representation of the test input (grid-level).
            train_input_objects_batch (list of lists of lists of dict): Extracted objects for training inputs (B, num_pairs, list_of_objects).
            train_output_objects_batch (list of lists of lists of dict): Extracted objects for training outputs (B, num_pairs, list_of_objects).
            test_input_objects_batch (list of lists of dict): Extracted objects for test input (B, list_of_objects).
            image_encoder_ref (ARCNeuralNetwork): Reference to the image encoder to use its _encode_object method.
            problem_batch_dims (dict): Contains 'train_input_dims', 'train_output_dims', 'test_input_dims' tensors.

        Returns:
            Tensor: A 'rule embedding' that captures the pattern.
        """
        batch_size = encoded_test_input.size(0)
        device = encoded_test_input.device

        # --- Helper for processing object lists through the object Transformer ---
        def process_objects_batch(object_list_for_grids_in_batch, grid_h_batch_for_objects, grid_w_batch_for_objects):
            """
            Processes a batch of object lists (variable length) through the object Transformer.
            Returns a summarized object embedding for each grid in the batch.
            object_list_for_grids_in_batch: list of (list of dict) for the batch (e.g., [obj_list_grid1, obj_list_grid2, ...])
            grid_h_batch_for_objects, grid_w_batch_for_objects: (B,) tensors of true dimensions for each grid.
            """
            all_obj_embeddings_padded = []
            attention_masks = [] # For variable length sequences

            for b_idx in range(batch_size):
                obj_list = object_list_for_grids_in_batch[b_idx]
                grid_h = grid_h_batch_for_objects[b_idx].item()
                grid_w = grid_w_batch_for_objects[b_idx].item()

                current_obj_embeddings = []
                # Add CLS token at the beginning of the sequence for aggregation
                current_obj_embeddings.append(self.object_cls_token.squeeze(0)) # Squeeze from (1, latent_dim) to (latent_dim)
                
                # Embed each object
                for obj_data in obj_list:
                    # Basic check for valid object data (e.g., non-empty, valid color/size)
                    if obj_data and 'color' in obj_data and obj_data['size'] > 0:
                        obj_emb = image_encoder_ref._encode_object(obj_data, grid_h, grid_w)
                        current_obj_embeddings.append(obj_emb)
                
                # Pad/truncate object embeddings for fixed sequence length
                num_actual_objects_plus_cls = len(current_obj_embeddings)
                
                padded_seq = torch.zeros(self.object_seq_len, self.latent_dim, device=device)
                
                # Fill with actual embeddings, truncate if too many
                if num_actual_objects_plus_cls > 0:
                    current_obj_embeddings_stacked = torch.stack(current_obj_embeddings[:self.object_seq_len])
                    padded_seq[:current_obj_embeddings_stacked.size(0)] = current_obj_embeddings_stacked

                # Create attention mask: False for attended, True for masked (padding)
                mask = torch.ones(self.object_seq_len, dtype=torch.bool, device=device)
                mask[:min(num_actual_objects_plus_cls, self.object_seq_len)] = False # Only mask elements that are actually present

                all_obj_embeddings_padded.append(padded_seq)
                attention_masks.append(mask)

            # Stack for batching (batch_size, object_seq_len, latent_dim)
            stacked_obj_embeddings = torch.stack(all_obj_embeddings_padded)
            stacked_attention_masks = torch.stack(attention_masks) # (batch_size, object_seq_len)

            # Add positional embeddings to the object embeddings
            stacked_obj_embeddings = stacked_obj_embeddings + self.object_positional_embeddings.unsqueeze(0)

            # Pass through object Transformer
            object_transformer_output = self.object_transformer_encoder(
                stacked_obj_embeddings,
                src_key_padding_mask=stacked_attention_masks
            ) # (batch_size, object_seq_len, latent_dim)

            # Extract the CLS token embedding as the summarized object-centric embedding for each grid
            summarized_object_embedding = object_transformer_output[:, 0, :] # CLS token is at index 0

            return summarized_object_embedding


        # --- Process each grid's objects to get summarized object embeddings ---

        # For train inputs:
        summarized_train_input_objects = []
        for i in range(self.num_train_pairs):
            summarized_train_input_objects.append(
                process_objects_batch(
                    train_input_objects_batch[i], # This is a list of object lists for the current pair
                    problem_batch_dims['train_input_dims'][i][:, 0], # Heights for this train input pair
                    problem_batch_dims['train_input_dims'][i][:, 1]  # Widths for this train input pair
                )
            )

        # For train outputs:
        summarized_train_output_objects = []
        for i in range(self.num_train_pairs):
            summarized_train_output_objects.append(
                process_objects_batch(
                    train_output_objects_batch[i],
                    problem_batch_dims['train_output_dims'][i][:, 0],
                    problem_batch_dims['train_output_dims'][i][:, 1]
                )
            )

        # For test input:
        summarized_test_input_objects = process_objects_batch(
            test_input_objects_batch, # This is a list of object lists for the test input
            problem_batch_dims['test_input_dims'][:, 0],
            problem_batch_dims['test_input_dims'][:, 1]
        )


        # --- Fuse grid-level and object-level embeddings ---
        # Each fused embedding is (batch_size, latent_dim)
        fused_train_inputs = []
        for i in range(self.num_train_pairs):
            combined = torch.cat([encoded_train_inputs[i], summarized_train_input_objects[i]], dim=1)
            fused_train_inputs.append(self.fusion_mlp(combined))

        fused_train_outputs = []
        for i in range(self.num_train_pairs):
            combined = torch.cat([encoded_train_outputs[i], summarized_train_output_objects[i]], dim=1)
            fused_train_outputs.append(self.fusion_mlp(combined))

        fused_test_input = self.fusion_mlp(torch.cat([encoded_test_input, summarized_test_input_objects], dim=1))


        # --- Main Transformer for rule inference (uses fused embeddings) ---
        sequence_elements = []
        # Create transformation embeddings for each train pair (using fused embeddings)
        for i in range(self.num_train_pairs):
            transformation_embedding = fused_train_outputs[i] - fused_train_inputs[i] # Delta of fused embeddings
            sequence_elements.append(transformation_embedding + self.positional_embeddings[i])

        # Add the fused test input embedding to the sequence
        test_input_embedding_with_pos = fused_test_input + self.positional_embeddings[self.num_train_pairs]
        sequence_elements.append(test_input_embedding_with_pos)

        stacked_embeddings = torch.stack(sequence_elements, dim=1)
        
        main_transformer_output = self.main_transformer_encoder(stacked_embeddings)

        rule_embedding = main_transformer_output[:, -1, :] # Typically CLS token (if present) or last element for rule

        return rule_embedding
    
class OutputDecoder(nn.Module):
    """
    Conceptual module to generate the output grid based on the inferred rule
    and the test input's latent representation.
    Now uses a conditioning MLP and Conditional Batch Normalization for adaptive generation.
    """
    def __init__(self, latent_dim=256, num_colors=10, grid_size=30):
        super(OutputDecoder, self).__init__()
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.initial_dim = grid_size // 2 # From ARCNeuralNetwork's conv3 stride=2

        # Input to decoder: concatenated rule embedding and test input latent representation
        self.combined_embedding_dim = latent_dim * 2

        # New: MLP to process the combined embedding for richer conditioning
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(self.combined_embedding_dim, latent_dim * 2),
            nn.GELU(), # Using GELU
            nn.Linear(latent_dim * 2, latent_dim * 2) # Output dimension for conditioning
        )
        self.decoder_input_dim = latent_dim * 2 # Output of conditioning_mlp

        # Layers for generating the output grid (fixed max size)
        # input_fc_grid takes the conditioned_embedding
        self.input_fc_grid = nn.Linear(self.decoder_input_dim, 256 * self.initial_dim * self.initial_dim)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_deconv1 = ConditionalBatchNorm2d(128, self.decoder_input_dim) # Changed to CBN
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_deconv2 = ConditionalBatchNorm2d(64, self.decoder_input_dim) # Changed to CBN
        self.deconv3 = nn.ConvTranspose2d(64, num_colors, kernel_size=3, stride=1, padding=1) # Output num_colors channels

        # Layers for predicting output height and width, also taking conditioned_embedding
        self.size_prediction_head_h = nn.Linear(self.decoder_input_dim, grid_size)
        self.size_prediction_head_w = nn.Linear(self.decoder_input_dim, grid_size)

    def forward(self, rule_embedding, encoded_test_input):
        """
        Args:
            rule_embedding (Tensor): The inferred rule embedding.
            encoded_test_input (Tensor): Latent representation of the test input.

        Returns:
            tuple:
                - predicted_output_logits (Tensor): Predicted output grid (logits for each color at each pixel).
                                                   Shape: (batch_size, num_colors, grid_size, grid_size)
                - predicted_height_logits (Tensor): Logits for predicted output height. Shape: (batch_size, grid_size)
                - predicted_width_logits (Tensor): Logits for predicted output width. Shape: (batch_size, grid_size)
        """
        # Concatenate rule embedding and test input embedding
        combined_embedding = torch.cat([rule_embedding, encoded_test_input], dim=1)

        # Process combined embedding through conditioning MLP
        conditioned_embedding = self.conditioning_mlp(combined_embedding) # (batch_size, decoder_input_dim)

        # --- Grid prediction path ---
        x_grid = F.gelu(self.input_fc_grid(conditioned_embedding)) # Using GELU
        x_grid = x_grid.view(x_grid.size(0), 256, self.initial_dim, self.initial_dim) # Reshape to (B, C, H, W)

        # Pass conditioning embedding to CBN layers
        x_grid = F.gelu(self.bn_deconv1(self.deconv1(x_grid), conditioned_embedding)) # Using GELU
        x_grid = F.gelu(self.bn_deconv2(self.deconv2(x_grid), conditioned_embedding)) # Using GELU
        predicted_output_logits = self.deconv3(x_grid) # No activation on final layer for logits

        # --- Size prediction path ---
        predicted_height_logits = self.size_prediction_head_h(conditioned_embedding)
        predicted_width_logits = self.size_prediction_head_w(conditioned_embedding)

        return predicted_output_logits, predicted_height_logits, predicted_width_logits    