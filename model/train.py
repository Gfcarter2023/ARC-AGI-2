import json
import warnings
import torch
import torch.optim as optim
warnings.filterwarnings('ignore')

def train_arc_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, evaluation_challenges,
                    val_dataset, GRID_SIZE):
    # New: Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, problem_batch in enumerate(train_loader):
            # Move all tensor data in the batch to the device
            problem_batch['test_input'] = problem_batch['test_input'].to(device)
            problem_batch['test_input_dims'] = problem_batch['test_input_dims'].to(device)
            # 'test_input_objects' is a list of lists of dicts (non-tensor), it remains on CPU
            # It's processed into tensors within ProblemEncoder.

            problem_batch['correct_test_output'] = problem_batch['correct_test_output'].to(device)
            problem_batch['correct_test_output_height'] = problem_batch['correct_test_output_height'].to(device)
            problem_batch['correct_test_output_width'] = problem_batch['correct_test_output_width'].to(device)
            # 'correct_test_output_objects' remains on CPU

            problem_batch['train_inputs'] = [t.to(device) for t in problem_batch['train_inputs']]
            problem_batch['train_input_dims'] = [t.to(device) for t in problem_batch['train_input_dims']]
            # 'train_input_objects' remains on CPU

            problem_batch['train_outputs'] = [t.to(device) for t in problem_batch['train_outputs']]
            problem_batch['train_output_dims'] = [t.to(device) for t in problem_batch['train_output_dims']]
            # 'train_output_objects' remains on CPU


            optimizer.zero_grad()

            # Forward pass through the full model
            predicted_output_logits, predicted_height_logits, predicted_width_logits = model(problem_batch)

            # --- Calculate Losses ---
            # 1. Grid Loss (masked)
            # Create a mask for the correct output region
            mask = torch.zeros_like(predicted_output_logits, dtype=torch.bool)
            for b in range(predicted_output_logits.size(0)):
                h_true = problem_batch['correct_test_output_height'][b].item()
                w_true = problem_batch['correct_test_output_width'][b].item()
                mask[b, :, :h_true, :w_true] = True

            # Apply mask to logits and target for loss calculation
            # Flatten for CrossEntropyLoss (which expects 2D inputs for N, C)
            masked_predicted_logits = predicted_output_logits.permute(0, 2, 3, 1)[mask.permute(0, 2, 3, 1)].view(-1,
                                                                                                                 predicted_output_logits.size(
                                                                                                                     1))
            masked_true_output = problem_batch['correct_test_output'][mask.permute(0, 2, 3, 1)[:, :, :, 0]].view(
                -1)  # Take first channel of mask

            # Only calculate grid loss if there are valid pixels
            if masked_true_output.numel() > 0:
                grid_loss = criterion(masked_predicted_logits, masked_true_output)
            else:
                grid_loss = torch.tensor(0.0, device=device)  # No valid pixels, no grid loss

            # 2. Size Loss (height and width as classification over 0 to GRID_SIZE-1)
            # Targets are 0-indexed, so we subtract 1 from true dimensions
            height_loss = criterion(predicted_height_logits, problem_batch['correct_test_output_height'] - 1)
            width_loss = criterion(predicted_width_logits, problem_batch['correct_test_output_width'] - 1)

            # Total loss
            LAMBDA_SIZE_LOSS = 0.1  # Hyperparameter to weigh size prediction loss
            loss = grid_loss + LAMBDA_SIZE_LOSS * (height_loss + width_loss)

            loss.backward()
            # New: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients to max_norm
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f} (Grid: {grid_loss.item():.4f}, H: {height_loss.item():.4f}, W: {width_loss.item():.4f})")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] finished. Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        total_pixel_accuracy = 0.0
        total_problem_correct = 0  # For problem-level exact match (grid + size)
        total_size_correct = 0  # For problems where height and width are correct
        incorrect_predictions_details = []  # To store differences for incorrect problems

        with torch.no_grad():
            for i, problem_batch in enumerate(val_loader):
                problem_batch['test_input'] = problem_batch['test_input'].to(device)
                problem_batch['test_input_dims'] = problem_batch['test_input_dims'].to(device)

                problem_batch['correct_test_output'] = problem_batch['correct_test_output'].to(device)
                problem_batch['correct_test_output_height'] = problem_batch['correct_test_output_height'].to(device)
                problem_batch['correct_test_output_width'] = problem_batch['correct_test_output_width'].to(device)

                problem_batch['train_inputs'] = [t.to(device) for t in problem_batch['train_inputs']]
                problem_batch['train_input_dims'] = [t.to(device) for t in problem_batch['train_input_dims']]

                problem_batch['train_outputs'] = [t.to(device) for t in problem_batch['train_outputs']]
                problem_batch['train_output_dims'] = [t.to(device) for t in problem_batch['train_output_dims']]


                predicted_output_logits, predicted_height_logits, predicted_width_logits = model(problem_batch)

                # --- Calculate Losses (similar to training for validation tracking) ---
                mask = torch.zeros_like(predicted_output_logits, dtype=torch.bool)
                for b in range(predicted_output_logits.size(0)):
                    h_true = problem_batch['correct_test_output_height'][b].item()
                    w_true = problem_batch['correct_test_output_width'][b].item()
                    mask[b, :, :h_true, :w_true] = True

                masked_predicted_logits = predicted_output_logits.permute(0, 2, 3, 1)[mask.permute(0, 2, 3, 1)].view(-1,
                                                                                                                     predicted_output_logits.size(
                                                                                                                         1))
                masked_true_output = problem_batch['correct_test_output'][mask.permute(0, 2, 3, 1)[:, :, :, 0]].view(-1)

                if masked_true_output.numel() > 0:
                    grid_loss = criterion(masked_predicted_logits, masked_true_output)
                else:
                    grid_loss = torch.tensor(0.0, device=device)

                height_loss = criterion(predicted_height_logits, problem_batch['correct_test_output_height'] - 1)
                width_loss = criterion(predicted_width_logits, problem_batch['correct_test_output_width'] - 1)
                loss = grid_loss + LAMBDA_SIZE_LOSS * (height_loss + width_loss)
                val_loss += loss.item()

                # --- Calculate Accuracies ---
                predicted_output_indices = torch.argmax(predicted_output_logits, dim=1)
                true_output_indices = problem_batch['correct_test_output']

                # Pixel-wise accuracy (only on the unpadded region)
                correct_pixels_batch = (predicted_output_indices[mask[:, 0, :, :]] == true_output_indices[
                    mask[:, 0, :, :]]).sum().item()
                total_pixels_in_batch = mask[:, 0, :, :].sum().item()  # Count only true pixels
                if total_pixels_in_batch > 0:
                    total_pixel_accuracy += correct_pixels_batch  # Accumulate total correct pixels

                # Predicted sizes
                predicted_height = torch.argmax(predicted_height_logits, dim=1) + 1  # Convert 0-indexed to 1-indexed
                predicted_width = torch.argmax(predicted_width_logits, dim=1) + 1

                # Problem-level accuracy (exact match of grid AND size)
                for b in range(predicted_output_indices.size(0)):
                    h_true = problem_batch['correct_test_output_height'][b].item()
                    w_true = problem_batch['correct_test_output_width'][b].item()

                    is_size_correct = (predicted_height[b].item() == h_true) and \
                                      (predicted_width[b].item() == w_true)

                    if is_size_correct:
                        total_size_correct += 1
                        # Check grid match only if size is correct
                        # Compare only the relevant region of the predicted grid
                        predicted_subgrid = predicted_output_indices[b, :h_true, :w_true]
                        true_subgrid = true_output_indices[b, :h_true, :w_true]
                        is_grid_correct = torch.equal(predicted_subgrid, true_subgrid)

                        if is_grid_correct:
                            total_problem_correct += 1

                    if not (is_size_correct and is_grid_correct):  # If problem is incorrect
                        incorrect_predictions_details.append({
                            'task_id': problem_batch['task_ids'][b],
                            'correct_output': true_output_indices[b].cpu().numpy().tolist(),
                            'predicted_output': predicted_output_indices[b].cpu().numpy().tolist(),  # Full 30x30 output
                            'correct_height': h_true,
                            'correct_width': w_true,
                            'predicted_height': predicted_height[b].item(),
                            'predicted_width': predicted_width[b].item()
                        })

        avg_val_loss = val_loss / len(val_loader)
        # Calculate average pixel accuracy over the entire validation set
        total_true_pixels_in_val_set = sum(
            p['correct_test_output_height'] * p['correct_test_output_width'] for p in val_dataset.problems if
            p['correct_test_output'] is not None)

        avg_pixel_accuracy = total_pixel_accuracy / total_true_pixels_in_val_set if total_true_pixels_in_val_set > 0 else 0.0
        problem_accuracy = total_problem_correct / len(val_dataset)
        size_accuracy = total_size_correct / len(val_dataset)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Pixel-wise Accuracy (on true region): {avg_pixel_accuracy:.4f}")
        print(f"Validation Size Accuracy (H & W correct): {size_accuracy:.4f}")
        print(f"Validation Problem-level Accuracy (grid & size exact match): {problem_accuracy:.4f}")

        if incorrect_predictions_details:
            print(f"Found {len(incorrect_predictions_details)} incorrect predictions in this epoch's validation set.")
            incorrect_filename = 'incorrect.json'
            with open(incorrect_filename, 'w') as f:
                json.dump(incorrect_predictions_details, f, indent=4)
            print(f"Incorrect saved to {incorrect_filename}")
        else:
            print("All validation problems predicted correctly in this epoch!")
        print("\n")
        
        # New: Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)


def predict_on_test_data(model, test_loader, device, GRID_SIZE=30):
    """
    Generates predictions for the test dataset and formats them for submission.
    """
    model.to(device)
    model.eval()
    all_predictions = {}
    with torch.no_grad():
        for i, problem_batch in enumerate(test_loader):
            # Move tensor data in the batch to device
            problem_batch['test_input'] = problem_batch['test_input'].to(device)
            problem_batch['test_input_dims'] = problem_batch['test_input_dims'].to(device)
            # 'test_input_objects' remains on CPU

            problem_batch['train_inputs'] = [t.to(device) for t in problem_batch['train_inputs']]
            problem_batch['train_input_dims'] = [t.to(device) for t in problem_batch['train_input_dims']]
            # 'train_input_objects' remains on CPU

            problem_batch['train_outputs'] = [t.to(device) for t in problem_batch['train_outputs']]
            problem_batch['train_output_dims'] = [t.to(device) for t in problem_batch['train_output_dims']]
            # 'train_output_objects' remains on CPU


            predicted_output_logits, predicted_height_logits, predicted_width_logits = model(problem_batch)

            # Get the predicted color indices for each pixel
            predicted_output_indices = torch.argmax(predicted_output_logits, dim=1)  # (B, H, W)

            # Get the predicted height and width
            predicted_height = torch.argmax(predicted_height_logits, dim=1) + 1  # Convert 0-indexed to 1-indexed
            predicted_width = torch.argmax(predicted_width_logits, dim=1) + 1

            # Convert predictions back to list-of-lists format for JSON
            for j in range(predicted_output_indices.size(0)):  # Iterate through batch
                task_id = problem_batch['task_ids'][j]

                # Crop the predicted output to its predicted dimensions
                h_pred = predicted_height[j].item()
                w_pred = predicted_width[j].item()

                # Ensure predicted dimensions are within valid range (1 to GRID_SIZE)
                h_pred = max(1, min(h_pred, GRID_SIZE))
                w_pred = max(1, min(w_pred, GRID_SIZE))

                # Slice the predicted grid
                predicted_grid_np = predicted_output_indices[j, :h_pred, :w_pred].cpu().numpy().tolist()

                # ARC submission format expects a list of outputs for each test case.
                # Since each task has only one test case, it's a list with one item.
                all_predictions[task_id] = [{'output': predicted_grid_np}]
    return all_predictions