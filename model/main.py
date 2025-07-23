import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import helper
from neural_network import FullARCModel
from dataset import ARCDataset, collate_fn
from train import train_arc_model, predict_on_test_data

warnings.filterwarnings('ignore')

 # Hyperparameters
NUM_COLORS = 10
GRID_SIZE = 30 # Max grid size
LATENT_DIM = 512
NUM_TRAIN_PAIRS = 2 # Number of (input, output) examples per problem
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the full model
full_arc_model = FullARCModel(
    num_colors=NUM_COLORS,
    grid_size=GRID_SIZE,
    latent_dim=LATENT_DIM,
    num_train_pairs=NUM_TRAIN_PAIRS
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # Used for both grid and size prediction
# New: Add weight_decay to Adam optimizer for L2 regularization
optimizer = optim.Adam(full_arc_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 

    # --- Load ARC Data ---
print("Loading ARC dataset files...")
try:
    training_solutions = json.load(open(helper.get_path('arc-agi_training_solutions.json')))
    evaluation_solutions = json.load(open(helper.get_path('arc-agi_evaluation_solutions.json')))
    evaluation_challenges = json.load(open(helper.get_path('arc-agi_evaluation_challenges.json')))
    training_challenges = json.load(open(helper.get_path('arc-agi_training_challenges.json')))
    test_challenges = json.load(open(helper.get_path('arc-agi_test_challenges.json')))
    print("ARC dataset files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please ensure 'get_path' is correctly configured and files exist.")
    exit()

# Create datasets and data loaders
print("Creating datasets and data loaders...")
# New: Pass augment=True to train_dataset for data augmentation
train_dataset = ARCDataset(training_challenges, training_solutions, NUM_COLORS, GRID_SIZE, NUM_TRAIN_PAIRS, augment=True)
eval_dataset = ARCDataset(evaluation_challenges, evaluation_solutions, NUM_COLORS, GRID_SIZE, NUM_TRAIN_PAIRS, augment=False) # No augmentation for validation
test_dataset = ARCDataset(test_challenges, {}, NUM_COLORS, GRID_SIZE, NUM_TRAIN_PAIRS, augment=False) # No augmentation for test

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
print(f"Datasets created. Train problems: {len(train_dataset)}, Evaluation problems: {len(eval_dataset)}, Test problems: {len(test_dataset)}")

# Start training
print("Starting training...")
train_arc_model(full_arc_model, train_loader, eval_loader, optimizer, criterion, NUM_EPOCHS, device, evaluation_challenges, eval_dataset, GRID_SIZE)
print("Training complete.")

# --- Generate Predictions for Test Data ---
print("\n--- Generating predictions for test data ---")
test_predictions = predict_on_test_data(full_arc_model, test_loader, device)

# Save predictions to a JSON file
submission_filename = 'data/output/submission.json'
with open(submission_filename, 'w') as f:
    json.dump(test_predictions, f, indent=4)
print(f"Predictions saved to {submission_filename}")