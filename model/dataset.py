import warnings
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
warnings.filterwarnings('ignore')

class ARCDataset(Dataset):
    def __init__(self, challenges, solutions, num_colors=10, grid_size=30, num_train_pairs=2, augment=False):
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.num_train_pairs = num_train_pairs
        self.augment = augment # New: flag to enable/disable augmentation
        self.problems = self._parse_arc_data(challenges, solutions)

    def _extract_connected_components(self, grid_data):
        """
        Extracts connected components (objects) from a 2D grid.
        Args:
            grid_data (list of lists of int): The raw grid data.
        Returns:
            list of dict: Each dict describes an object with 'color', 'pixels', 'bbox', 'size'.
        """
        if not grid_data or not grid_data[0]:
            return []

        rows, cols = len(grid_data), len(grid_data[0])
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []

        for r in range(rows):
            for c in range(cols):
                if not visited[r, c]:
                    current_color = grid_data[r][c]
                    if current_color == 0: # Assuming color 0 (black) is background/empty, ignore
                        visited[r, c] = True
                        continue

                    # Start BFS/DFS for connected component
                    component_pixels = []
                    q = [(r, c)]
                    visited[r, c] = True
                    min_r, max_r = r, r
                    min_c, max_c = c, c

                    head = 0
                    while head < len(q):
                        curr_r, curr_c = q[head]
                        head += 1
                        component_pixels.append((curr_r, curr_c))

                        min_r = min(min_r, curr_r)
                        max_r = max(max_r, curr_r)
                        min_c = min(min_c, curr_c)
                        max_c = max(max_c, curr_c)

                        # Check 4-directional neighbors
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid_data[nr][nc] == current_color:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                    if component_pixels:
                        objects.append({
                            'color': current_color,
                            'pixels': component_pixels,
                            'bbox': (min_r, min_c, max_r, max_c),
                            'size': len(component_pixels)
                        })
        return objects

    def _process_grid(self, grid_data):
        """
        Converts a list-of-lists grid from JSON into a one-hot encoded PyTorch tensor.
        Pads smaller grids to GRID_SIZE x GRID_SIZE.
        Returns the one-hot encoded grid, its original height, and its original width.
        """
        if not grid_data:  # Handle empty grids gracefully
            return self._create_empty_grid(), 0, 0 # Return 0,0 for dimensions of empty grid

        grid_h, grid_w = len(grid_data), len(grid_data[0])

        # Create a tensor for the current grid, padded with 0s
        padded_grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)

        # Fill the padded grid with actual data
        # Only copy up to the defined GRID_SIZE, ignoring any extra if grid_data is larger
        for r in range(min(grid_h, self.grid_size)):
            for c in range(min(grid_w, self.grid_size)):
                # Ensure the element being assigned is an integer
                if isinstance(grid_data[r][c], list):
                    # This implies deeper nesting than expected, taking first element
                    print(f"Warning: grid_data[{r}][{c}] is a list. Taking the first element. Data: {grid_data[r][c]}")
                    padded_grid[r, c] = grid_data[r][c][0]
                else:
                    padded_grid[r, c] = grid_data[r][c]

        # One-hot encode the grid. Output shape: (num_colors, grid_size, grid_size)
        one_hot_grid = F.one_hot(padded_grid, num_classes=self.num_colors).permute(2, 0, 1).float()
        return one_hot_grid, grid_h, grid_w # Return original dimensions

    def _create_empty_grid(self):
        """Creates an empty 30x30 grid (all color 0), one-hot encoded."""
        empty_grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        return F.one_hot(empty_grid, num_classes=self.num_colors).permute(2, 0, 1).float()

    def _augment_pair(self, input_grid_tensor, input_h, input_w, input_objects,
                      output_grid_tensor, output_h, output_w, output_objects):
        """
        Applies random geometric and color transformations to an input-output pair.
        Transformations are applied consistently to both grids and their associated objects/dimensions.
        """
        # 1. Geometric Transformations (Rotation, Flip)
        rotation_type = random.choice([0, 1, 2, 3]) # 0, 90, 180, 270 degrees
        flip_h = random.choice([True, False]) # Horizontal flip
        flip_v = random.choice([True, False]) # Vertical flip

        # Apply to input grid tensor
        aug_input_grid = input_grid_tensor
        if flip_h:
            aug_input_grid = torch.flip(aug_input_grid, dims=[2]) # Flip along width (dim 2)
        if flip_v:
            aug_input_grid = torch.flip(aug_input_grid, dims=[1]) # Flip along height (dim 1)
        aug_input_grid = torch.rot90(aug_input_grid, k=rotation_type, dims=[1, 2]) # Rotate

        # Apply to output grid tensor
        aug_output_grid = output_grid_tensor
        if flip_h:
            aug_output_grid = torch.flip(aug_output_grid, dims=[2])
        if flip_v:
            aug_output_grid = torch.flip(aug_output_grid, dims=[1])
        aug_output_grid = torch.rot90(aug_output_grid, k=rotation_type, dims=[1, 2])

        # Update dimensions based on rotation
        aug_input_h, aug_input_w = input_h, input_w
        aug_output_h, aug_output_w = output_h, output_w
        if rotation_type % 2 == 1: # 90 or 270 degrees rotation swaps height and width
            aug_input_h, aug_input_w = input_w, input_h
            aug_output_h, aug_output_w = output_w, output_h

        # 2. Color Permutation
        # Create a random mapping for colors (excluding black/0)
        colors_to_permute = list(range(1, self.num_colors))
        random.shuffle(colors_to_permute)
        color_map = {i+1: colors_to_permute[i] for i in range(len(colors_to_permute))}
        color_map[0] = 0 # Black stays black

        # Apply color permutation to one-hot encoded grids
        # Convert one-hot to index grid, apply map, convert back to one-hot
        # Detach and clone to avoid modifying original tensor being processed
        aug_input_grid_indices = torch.argmax(aug_input_grid, dim=0).clone().detach() # (H, W)
        aug_output_grid_indices = torch.argmax(aug_output_grid, dim=0).clone().detach() # (H, W)

        # Apply color map manually for each pixel
        for r in range(aug_input_grid_indices.shape[0]):
            for c in range(aug_input_grid_indices.shape[1]):
                aug_input_grid_indices[r, c] = color_map.get(aug_input_grid_indices[r, c].item(), aug_input_grid_indices[r, c].item())
                aug_output_grid_indices[r, c] = color_map.get(aug_output_grid_indices[r, c].item(), aug_output_grid_indices[r, c].item())

        # Convert back to one-hot
        aug_input_grid = F.one_hot(aug_input_grid_indices, num_classes=self.num_colors).permute(2, 0, 1).float()
        aug_output_grid = F.one_hot(aug_output_grid_indices, num_classes=self.num_colors).permute(2, 0, 1).float()

        # Update object properties (pixels, bbox, color) based on augmentations
        # Pass aug_input_h/w etc. for correct normalization in _augment_objects
        aug_input_objects = self._augment_objects(input_objects, input_h, input_w, rotation_type, flip_h, flip_v, color_map)
        aug_output_objects = self._augment_objects(output_objects, output_h, output_w, rotation_type, flip_h, flip_v, color_map)

        return aug_input_grid, aug_input_h, aug_input_w, aug_input_objects, \
               aug_output_grid, aug_output_h, aug_output_w, aug_output_objects

    def _augment_objects(self, objects, original_h, original_w, rotation_type, flip_h, flip_v, color_map):
        """
        Applies geometric and color transformations to a list of object dictionaries.
        Consistent with _augment_grid.
        """
        augmented_objects = []
        for obj in objects:
            aug_pixels = []
            
            # Initialize min/max with values that will be updated by first pixel
            min_r_aug, max_r_aug = self.grid_size, -1
            min_c_aug, max_c_aug = self.grid_size, -1

            for r_orig, c_orig in obj['pixels']:
                # Apply flips
                if flip_h:
                    c_orig = original_w - 1 - c_orig
                if flip_v:
                    r_orig = original_h - 1 - r_orig

                # Apply rotation
                if rotation_type == 1: # 90 deg clockwise
                    r_new, c_new = c_orig, original_h - 1 - r_orig
                elif rotation_type == 2: # 180 deg
                    r_new, c_new = original_h - 1 - r_orig, original_w - 1 - c_orig
                elif rotation_type == 3: # 270 deg clockwise
                    r_new, c_new = original_w - 1 - c_orig, r_orig
                else: # 0 deg
                    r_new, c_new = r_orig, c_orig

                aug_pixels.append((r_new, c_new))
                min_r_aug = min(min_r_aug, r_new)
                max_r_aug = max(max_r_aug, r_new)
                min_c_aug = min(min_c_aug, c_new)
                max_c_aug = max(max_c_aug, c_new)

            # Apply color permutation
            aug_color = color_map.get(obj['color'], obj['color'])

            if not aug_pixels: # Handle cases where object might become empty after filtering/edge cases
                augmented_objects.append({
                    'color': aug_color,
                    'pixels': [],
                    'bbox': (0, 0, -1, -1), # Invalid bbox for empty object
                    'size': 0
                })
            else:
                augmented_objects.append({
                    'color': aug_color,
                    'pixels': aug_pixels,
                    'bbox': (min_r_aug, min_c_aug, max_r_aug, max_c_aug),
                    'size': obj['size'] # Size remains the same after geometric transforms
                })
        return augmented_objects


    def _parse_arc_data(self, challenges, solutions):
        """
        Parses ARC challenge and solution data into a format suitable for the model.
        Now also stores original input/output dimensions for all grids and extracted objects.
        """
        problems = []
        for task_id, challenge_data in challenges.items():
            task_solution_raw = solutions.get(task_id, None)

            # Extract training pairs
            train_inputs = []
            train_outputs = []
            train_input_dims = []
            train_output_dims = []
            train_input_objects = [] # New: store extracted objects for train inputs
            train_output_objects = [] # New: store extracted objects for train outputs

            for pair in challenge_data['train']:
                # Store raw grid data for augmentation later
                raw_input_grid = pair['input']
                raw_output_grid = pair['output']

                processed_input_tensor, h_in, w_in = self._process_grid(raw_input_grid)
                processed_output_tensor, h_out, w_out = self._process_grid(raw_output_grid)
                
                train_inputs.append(processed_input_tensor)
                train_outputs.append(processed_output_tensor)
                train_input_dims.append((h_in, w_in))
                train_output_dims.append((h_out, w_out))
                
                # Extract objects from the original raw grid data
                train_input_objects.append(self._extract_connected_components(raw_input_grid))
                train_output_objects.append(self._extract_connected_components(raw_output_grid))


            # Pad or truncate training pairs to NUM_TRAIN_PAIRS
            while len(train_inputs) < self.num_train_pairs:
                train_inputs.append(self._create_empty_grid())
                train_outputs.append(self._create_empty_grid())
                train_input_dims.append((0, 0)) # Pad dimensions as well for consistency
                train_output_dims.append((0, 0))
                train_input_objects.append([]) # Pad with empty object list
                train_output_objects.append([]) # Pad with empty object list

            train_inputs = train_inputs[:self.num_train_pairs]
            train_outputs = train_outputs[:self.num_train_pairs]
            train_input_dims = train_input_dims[:self.num_train_pairs]
            train_output_dims = train_output_dims[:self.num_train_pairs]
            train_input_objects = train_input_objects[:self.num_train_pairs]
            train_output_objects = train_output_objects[:self.num_train_pairs]


            # Extract test input (assuming there's always at least one test example)
            raw_test_input_grid = challenge_data['test'][0]['input']
            test_input_grid_tensor, test_input_h, test_input_w = self._process_grid(raw_test_input_grid)
            test_input_objects = self._extract_connected_components(raw_test_input_grid)

            correct_test_output = None
            correct_test_output_height = None
            correct_test_output_width = None
            correct_test_output_objects = None # New

            if task_solution_raw is not None and isinstance(task_solution_raw, list) and len(task_solution_raw) > 0:
                actual_solution_grid = task_solution_raw[0]
                if isinstance(actual_solution_grid, list) and len(actual_solution_grid) > 0 and isinstance(
                        actual_solution_grid[0], list):
                    # Store original dimensions of the correct output
                    correct_test_output_height = len(actual_solution_grid)
                    correct_test_output_width = len(actual_solution_grid[0])

                    # Process grid (will pad to GRID_SIZE for consistency in tensor shape)
                    processed_output_grid, _, _ = self._process_grid(actual_solution_grid) # Don't need dims here again
                    correct_test_output = torch.argmax(processed_output_grid,
                                                       dim=0)  # Convert to color indices for CE loss
                    correct_test_output_objects = self._extract_connected_components(actual_solution_grid)

            problems.append({
                'task_id': task_id,
                'train_inputs': train_inputs,
                'train_outputs': train_outputs,
                'train_input_dims': train_input_dims,
                'train_output_dims': train_output_dims,
                'train_input_objects': train_input_objects,
                'train_output_objects': train_output_objects,
                'test_input': test_input_grid_tensor,
                'test_input_dims': (test_input_h, test_input_w),
                'test_input_objects': test_input_objects,
                'correct_test_output': correct_test_output,  # Will be None for test set without solutions
                'correct_test_output_height': correct_test_output_height,
                'correct_test_output_width': correct_test_output_width,
                'correct_test_output_objects': correct_test_output_objects # New
            })
        return problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]

        # Clone tensors for augmentation to avoid modifying the original data stored in self.problems
        train_inputs = [t.clone() for t in problem['train_inputs']]
        train_outputs = [t.clone() for t in problem['train_outputs']]
        # Dimensions are tuples, so simple copy is fine for mutable list of tuples
        train_input_dims = list(problem['train_input_dims'])
        train_output_dims = list(problem['train_output_dims'])
        # Object lists are lists of dicts, which are mutable. If we augment in place, we need deepcopy or re-creation
        # _augment_pair will re-create them, so shallow copy here is fine for the list of lists
        train_input_objects = [list(obj_list) for obj_list in problem['train_input_objects']]
        train_output_objects = [list(obj_list) for obj_list in problem['train_output_objects']]

        test_input = problem['test_input'].clone()
        test_input_dims = problem['test_input_dims'] # This is already a tuple (h, w)
        test_input_objects = list(problem['test_input_objects'])

        correct_test_output = problem['correct_test_output'] # This is an index grid (H,W)
        correct_test_output_height = problem['correct_test_output_height']
        correct_test_output_width = problem['correct_test_output_width']
        correct_test_output_objects = problem['correct_test_output_objects']

        if self.augment:
            # Apply augmentation to each train pair
            for i in range(self.num_train_pairs):
                # Call _augment_pair and capture all its return values into temporary variables
                aug_input_grid, aug_input_h, aug_input_w, aug_input_objects, \
                aug_output_grid, aug_output_h, aug_output_w, aug_output_objects = \
                    self._augment_pair(train_inputs[i], train_input_dims[i][0], train_input_dims[i][1], train_input_objects[i],
                                       train_outputs[i], train_output_dims[i][0], train_output_dims[i][1], train_output_objects[i])
                
                # Now, assign the augmented values back to the list elements.
                # For dimensions, assign a new tuple to the list element.
                train_inputs[i] = aug_input_grid
                train_input_dims[i] = (aug_input_h, aug_input_w) 
                train_input_objects[i] = aug_input_objects

                train_outputs[i] = aug_output_grid
                train_output_dims[i] = (aug_output_h, aug_output_w) 
                train_output_objects[i] = aug_output_objects

            # Apply augmentation to the test input (and correct output if available)
            if correct_test_output is not None:
                # Convert target to one-hot for augmentation, then back to index grid
                correct_test_output_one_hot = F.one_hot(correct_test_output, num_classes=self.num_colors).permute(2,0,1).float()
                
                # Call _augment_pair and capture all its return values into temporary variables
                aug_test_input, aug_test_input_h, aug_test_input_w, aug_test_input_objects, \
                aug_correct_test_output_tensor, aug_correct_test_output_h, aug_correct_test_output_w, aug_correct_test_output_objects = \
                    self._augment_pair(test_input, test_input_dims[0], test_input_dims[1], test_input_objects,
                                       correct_test_output_one_hot,
                                       correct_test_output_height, correct_test_output_width, correct_test_output_objects)
                
                # Assign augmented values back.
                # For test_input_dims, assign a new tuple to the variable itself.
                test_input = aug_test_input
                test_input_dims = (aug_test_input_h, aug_test_input_w) 
                test_input_objects = aug_test_input_objects

                correct_test_output = torch.argmax(aug_correct_test_output_tensor, dim=0) # Convert back to index grid
                correct_test_output_height = aug_correct_test_output_h
                correct_test_output_width = aug_correct_test_output_w
                correct_test_output_objects = aug_correct_test_output_objects

            else: # If no correct output (e.g., for test set), just augment the test input
                # Call _augment_pair and capture all its return values
                aug_test_input, aug_test_input_h, aug_test_input_w, aug_test_input_objects, \
                _, _, _, _ = \
                    self._augment_pair(test_input, test_input_dims[0], test_input_dims[1], test_input_objects,
                                       self._create_empty_grid(), 0, 0, []) # Pass dummy output data
                
                # Assign augmented values back.
                # For test_input_dims, assign a new tuple to the variable itself.
                test_input = aug_test_input
                test_input_dims = (aug_test_input_h, aug_test_input_w) 
                test_input_objects = aug_test_input_objects


        item_dict = {
            'task_id': problem['task_id'],
            'train_inputs': train_inputs,
            'train_outputs': train_outputs,
            'train_input_dims': train_input_dims,
            'train_output_dims': train_output_dims,
            'train_input_objects': train_input_objects,
            'train_output_objects': train_output_objects,
            'test_input': test_input,
            'test_input_dims': test_input_dims,
            'test_input_objects': test_input_objects,
        }

        # Only include 'correct_test_output' and its dimensions/objects if they are not None
        if correct_test_output is not None:
            item_dict['correct_test_output'] = correct_test_output
            item_dict['correct_test_output_height'] = correct_test_output_height
            item_dict['correct_test_output_width'] = correct_test_output_width
            item_dict['correct_test_output_objects'] = correct_test_output_objects

        return item_dict
    
def collate_fn(batch):
    test_input_batch = torch.stack([item['test_input'] for item in batch])
    test_input_dims_batch = torch.tensor([item['test_input_dims'] for item in batch], dtype=torch.long)
    test_input_objects_batch = [item['test_input_objects'] for item in batch] # This is a list of lists of dicts

    num_train_pairs = len(batch[0]['train_inputs'])
    train_inputs_batch = [
        torch.stack([item['train_inputs'][i] for item in batch])
        for i in range(num_train_pairs)
    ]
    train_outputs_batch = [
        torch.stack([item['train_outputs'][i] for item in batch])
        for i in range(num_train_pairs)
    ]
    train_input_dims_batch = [
        torch.tensor([item['train_input_dims'][i] for item in batch], dtype=torch.long)
        for i in range(num_train_pairs)
    ]
    train_output_dims_batch = [
        torch.tensor([item['train_output_dims'][i] for item in batch], dtype=torch.long)
        for i in range(num_train_pairs)
    ]
    # Collect train input/output objects. These are lists of lists of lists of dicts (num_pairs, batch_size, list_of_objects)
    train_input_objects_batch = [
        [item['train_input_objects'][i] for item in batch]
        for i in range(num_train_pairs)
    ]
    train_output_objects_batch = [
        [item['train_output_objects'][i] for item in batch]
        for i in range(num_train_pairs)
    ]


    correct_test_output_batch = None
    correct_test_output_height_batch = None
    correct_test_output_width_batch = None
    correct_test_output_objects_batch = None

    if 'correct_test_output' in batch[0]:
        correct_test_output_batch = torch.stack([item['correct_test_output'] for item in batch])
        correct_test_output_height_batch = torch.tensor([item['correct_test_output_height'] for item in batch],
                                                        dtype=torch.long)
        correct_test_output_width_batch = torch.tensor([item['correct_test_output_width'] for item in batch],
                                                       dtype=torch.long)
        correct_test_output_objects_batch = [item['correct_test_output_objects'] for item in batch]

    task_ids = [item['task_id'] for item in batch]

    return {
        'task_ids': task_ids,
        'train_inputs': train_inputs_batch,
        'train_outputs': train_outputs_batch,
        'train_input_dims': train_input_dims_batch,
        'train_output_dims': train_output_dims_batch,
        'train_input_objects': train_input_objects_batch,
        'train_output_objects': train_output_objects_batch,
        'test_input': test_input_batch,
        'test_input_dims': test_input_dims_batch,
        'test_input_objects': test_input_objects_batch,
        'correct_test_output': correct_test_output_batch,
        'correct_test_output_height': correct_test_output_height_batch,
        'correct_test_output_width': correct_test_output_width_batch,
        'correct_test_output_objects': correct_test_output_objects_batch
    }