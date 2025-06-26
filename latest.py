import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import gc
import os
import random

# CUDA optimization settings
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def initialize_model(model_name, activation_name, num_classes):
    """
    Initialize a model with the specified architecture and activation function
    
    Args:
        model_name: Name of the model architecture
        activation_name: Name of the activation function
        num_classes: Number of output classes
        
    Returns:
        model: Initialized PyTorch model
    """
    print(f"Initializing {model_name} with {activation_name} activation...")
    
    # Define activation function based on name
    def get_activation(name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'lrelu':
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif name == 'prelu':
            return nn.PReLU()
        elif name == 'mish':
            return nn.Mish(inplace=True)
        elif name == 'swish' or name == 'silu':
            return nn.SiLU()  # SiLU is the same as Swish
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)  # Default to ReLU
    
    # Initialize model based on architecture name
    if model_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
        # Replace all ReLU activations with the specified activation
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name.split('.')[-1], get_activation(activation_name))
        # Modify the final layer for the number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                setattr(model, name.split('.')[-1], get_activation(activation_name))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        # For DenseNet, we need a different approach to replace activations
        # as they're part of the _modules dictionary
        for name, module in model.features.named_modules():
            if isinstance(module, nn.ReLU):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model.features
                    for part in parent_name.split('.'):
                        parent = parent._modules[part]
                    parent._modules[child_name] = get_activation(activation_name)
                else:
                    model.features._modules[child_name] = get_activation(activation_name)
        # Modify the classifier
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        # Replace activations in MobileNetV2
        for module in model.modules():
            if hasattr(module, 'activation'):
                module.activation = get_activation(activation_name)
        # Modify the classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    

    elif model_name == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        # Replace activations in VGG
        for i, module in enumerate(model.features):
            if isinstance(module, nn.ReLU):
                if activation_name == 'prelu':
                    # Use out_channels for each conv layer, else fallback to 1
                    prev = model.features[i-1]
                    if isinstance(prev, nn.Conv2d):
                        num_parameters = prev.out_channels
                    else:
                        num_parameters = 1
                    model.features[i] = nn.PReLU(num_parameters=num_parameters)
                else:
                    model.features[i] = get_activation(activation_name)
        # Modify classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'vgg19':
        model = models.vgg19(weights='IMAGENET1K_V1')
        for i, module in enumerate(model.features):
            if isinstance(module, nn.ReLU):
                if activation_name == 'prelu':
                    prev = model.features[i-1]
                    if isinstance(prev, nn.Conv2d):
                        num_parameters = prev.out_channels
                    else:
                        num_parameters = 1
                    model.features[i] = nn.PReLU(num_parameters=num_parameters)
                else:
                    model.features[i] = get_activation(activation_name)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
  
    elif model_name.startswith('efficientnet'):
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        else:
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            
        # For EfficientNet, we need to replace the activation in each MBConv block
        for module in model.modules():
            if hasattr(module, 'activation'):
                module.activation = get_activation(activation_name)
        # Modify the classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    # Move model to device and make it memory efficient
    model = model.to(device)
    
    # Initialize weights for the new layers
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model

def evaluate_model(model, dataloader, class_names, device=None):
    """
    Efficiently evaluate a model on the given dataloader
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing evaluation data
        class_names: List of class names for the confusion matrix
        device: Device to run evaluation on (defaults to CUDA if available)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    # Track time for inference performance measurement
    total_time = 0
    total_samples = 0
    
    # Use mixed precision for inference if available
    use_amp = torch.cuda.is_available()
    
    # Perform evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            # Measure batch processing time
            start_time = time.time()
            
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Run inference
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Track timing
            batch_time = time.time() - start_time
            total_time += batch_time
            total_samples += inputs.size(0)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Free memory
            del inputs, labels, outputs, preds
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Calculate average inference time per sample and throughput
    avg_time_per_sample = total_time / total_samples
    throughput = total_samples / total_time
    
    # Return comprehensive evaluation metrics
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'avg_time_per_sample': avg_time_per_sample,
        'throughput': throughput,
        'total_eval_time': total_time
    }

def visualize_evaluation_results(eval_results, model_name, activation_name, class_names, save_dir="figures"):
    """
    Visualize the evaluation results with plots and save them
    
    Args:
        eval_results: Dictionary containing evaluation metrics
        model_name: Name of the model
        activation_name: Name of the activation function
        class_names: List of class names
    """
    os.makedirs(save_dir, exist_ok=True)
    # Print summary metrics
    print(f"\n=== Evaluation Results for {model_name} with {activation_name} ===")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Average inference time: {eval_results['avg_time_per_sample']*1000:.2f} ms per sample")
    print(f"Throughput: {eval_results['throughput']:.2f} samples/second")
    
    # Print classification report
    print("\nClassification Report:")
    report_df = pd.DataFrame(eval_results['classification_report']).transpose()
    print(report_df.round(4))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        eval_results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name} with {activation_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}_{activation_name}.png"))
    # plt.show()  # Commented out to avoid blocking
    plt.close()
    
    # Plot per-class metrics
    plt.figure(figsize=(12, 6))
    metrics = pd.DataFrame(eval_results['classification_report']).transpose()
    metrics = metrics.iloc[:-3]  # Remove the aggregate metrics
    
    # Create per-class metric plot
    ax = metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
    plt.title(f'Per-Class Metrics - {model_name} with {activation_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"per_class_metrics_{model_name}_{activation_name}.png"))
    # plt.show()  # Commented out to avoid blocking
    plt.close()

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10, grad_clip=1.0, early_stopping_patience=5, save_best_path=None):
    """
    Train the model with the provided parameters, early stopping, and checkpointing
    
    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        num_epochs: Number of epochs to train for
        grad_clip: Maximum norm for gradient clipping
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        save_best_path: Path to save the best model checkpoint
        
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Use mixed precision for training if available
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Get initial GPU memory usage
    if torch.cuda.is_available():
        start_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory usage: {start_mem:.2f} MB")
    
    # Set up gradient accumulation for larger effective batch sizes
    # if needed (helpful for large models with memory constraints)
    gradient_accumulation_steps = 1  # Increase if needed for memory constraints
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
                # Clear cache before validation to free up memory
                clear_gpu_cache()
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Zero the parameter gradients (only at appropriate accumulation step)
                if phase == 'train' and batch_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            # Scale loss for gradient accumulation if using it
                            if gradient_accumulation_steps > 1:
                                loss = loss / gradient_accumulation_steps
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train' and gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        if use_amp:
                            scaler.scale(loss).backward()
                            
                            # Step optimizer only at appropriate accumulation step
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                # Gradient clipping
                                if grad_clip > 0:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                                
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            loss.backward()
                            
                            # Step optimizer only at appropriate accumulation step
                            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                                # Gradient clipping
                                if grad_clip > 0:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                                
                                optimizer.step()
                
                # Statistics - multiply loss by gradient_accumulation_steps to get true loss
                if gradient_accumulation_steps > 1 and phase == 'train':
                    running_loss += (loss.item() * gradient_accumulation_steps) * inputs.size(0)
                else:
                    running_loss += loss.item() * inputs.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Free up tensors from this batch immediately
                del inputs, labels, outputs, preds
                if phase == 'train' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': running_corrects.double().item() / total_samples
                })
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Early stopping and checkpointing
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_epoch = epoch
                    patience_counter = 0
                    if save_best_path:
                        torch.save(model.state_dict(), save_best_path)
                else:
                    patience_counter += 1
        
        # Step the scheduler after validation
        if hasattr(scheduler, 'step'):
            if hasattr(scheduler, 'optimizer') and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
        
        # Print memory usage at the end of each epoch
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1024**2
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"GPU memory: current={current_mem:.2f} MB, peak={max_mem:.2f} MB")
            # Clear cache between epochs
            clear_gpu_cache()
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
            break
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Load best model if checkpointing was used
    if save_best_path and os.path.exists(save_best_path):
        model.load_state_dict(torch.load(save_best_path))
    
    return model, history

def prepare_data(data_dir, batch_size=32):
    """
    Prepare data loaders for training and evaluation with extra augmentation
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for dataloaders
        
    Returns:
        dataloaders: Dictionary containing 'train', 'val', and 'test' dataloaders
        dataset_sizes: Dictionary containing the size of each dataset
        class_names: List of class names
    """
    # Data transforms with GPU acceleration where possible
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # Optional: Mixup/Cutmix can be added here
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    image_datasets = {
        split: datasets.ImageFolder(f"{data_dir}/{split}", data_transforms[split])
        for split in ['train', 'val', 'test']
    }
    
    # Calculate optimal number of workers for dataloaders
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 workers
    
    # Use persistent workers to avoid the overhead of worker creation
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=(split == 'train')  # Avoid incomplete batches during training
        )
        for split in ['train', 'val', 'test']
    }
    
    dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")
    print(f"Using {num_workers} dataloader workers")
    
    return dataloaders, dataset_sizes, class_names

def main():
    set_seed(42)
    # Configuration
    data_dir = "potato_split"  # Update this path
    batch_size = 64  # Increased batch size for better GPU utilization
    num_epochs = 20
    
    # Auto-tune batch size based on available GPU memory (if using CUDA)
    if torch.cuda.is_available():
        # Start with a reasonable batch size and try to maximize it
        # This is a simple heuristic - for production you might want a more sophisticated approach
        memory_available = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Total GPU memory: {memory_available:.2f} MB")
        
        # Use a larger batch size for models with smaller memory footprint
        if memory_available > 8000:  # >8GB memory
            batch_size = 128
        elif memory_available > 4000:  # >4GB memory
            batch_size = 64
        else:
            batch_size = 32
            
        print(f"Auto-tuned batch size: {batch_size}")
    
    # Define models to evaluate
    model_names = [
        'resnet18', 'resnet50', 'densenet121', 'mobilenet_v2', 
        'vgg16', 'vgg19', 'efficientnet_b0', 'efficientnet_b3'
    ]
    
    # Define activation functions to try
    activation_names = ['relu', 'lrelu', 'prelu', 'mish', 'swish', 'gelu']
    
    # For demonstration, we can use a subset
    selected_models = [
        'resnet18', 'resnet50', 'densenet121', 'mobilenet_v2', 
        'vgg16', 'vgg19', 'efficientnet_b0', 'efficientnet_b3'
    ]  # You can use all from model_names
    selected_activations = ['relu', 'lrelu', 'prelu', 'mish', 'swish', 'gelu']  # You can use all from activation_names
    
    # Print CUDA optimization status
    if torch.cuda.is_available():
        print(f"CUDA optimizations:")
        print(f"- cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"- TF32 allowed: {torch.backends.cudnn.allow_tf32}")
        print(f"- AMP enabled: True")
    
    # Prepare data
    dataloaders, dataset_sizes, class_names = prepare_data(data_dir, batch_size)
    num_classes = len(class_names)
    
    # Dictionary to store results
    results = {}
    
    # Train selected models with selected activation functions
    for model_name in selected_models:
        for activation_name in selected_activations:
            print(f"\nTraining {model_name} with {activation_name} activation")
            key = f"{model_name}_{activation_name}"
            
            # Clear GPU cache before initializing a new model
            clear_gpu_cache()
            
            # Initialize the model
            model = initialize_model(model_name, activation_name, num_classes)
            
            # Use model-specific learning rates for better convergence
            if 'efficientnet' in model_name:
                base_lr = 0.0005  # EfficientNet typically needs lower learning rates
            elif 'resnet' in model_name or 'densenet' in model_name:
                base_lr = 0.001
            elif 'vgg' in model_name:
                base_lr = 0.0001  # VGG typically requires smaller learning rates
            else:
                base_lr = 0.001
                
            # Define loss function with label smoothing for regularization
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Optimizers with appropriate parameters
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=base_lr,
                weight_decay=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Use ReduceLROnPlateau for better accuracy
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
            
            save_best_path = f"best_{model_name}_{activation_name}.pt"
            
            # Train the model with gradient clipping for stability
            trained_model, history = train_model(
                model, 
                criterion, 
                optimizer, 
                scheduler,
                dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
                num_epochs=num_epochs,
                grad_clip=1.0,
                early_stopping_patience=5,
                save_best_path=save_best_path
            )
            
            # Save the results
            results[key] = {
                'model': trained_model,
                'history': history
            }
            
            # Evaluate on test set
            eval_results = evaluate_model(trained_model, dataloaders['test'], class_names, device)
            results[key]['evaluation'] = eval_results
            
            # Visualize results
            visualize_evaluation_results(eval_results, model_name, activation_name, class_names)
            
            # Save and close training history plots
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Validation')
            plt.title(f'Loss - {model_name} with {activation_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Validation')
            plt.title(f'Accuracy - {model_name} with {activation_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", f"history_{model_name}_{activation_name}.png"))
            # plt.show()  # Commented out to avoid blocking
            plt.close()
    
    # Compare all models
    compare_models(results, selected_models, selected_activations)

def compare_models(results, model_names, activation_names):
    """
    Compare all trained models based on various metrics
    
    Args:
        results: Dictionary containing results for all models
        model_names: List of model names
        activation_names: List of activation names
    """
    # Extract metrics for comparison
    comparison_data = []
    
    for model_name in model_names:
        for activation_name in activation_names:
            key = f"{model_name}_{activation_name}"
            if key in results:
                eval_results = results[key]['evaluation']
                comparison_data.append({
                    'model': model_name,
                    'activation': activation_name,
                    'accuracy': eval_results['accuracy'],
                    'avg_time_per_sample': eval_results['avg_time_per_sample'],
                    'throughput': eval_results['throughput'],
                    'f1_score': eval_results['classification_report']['macro avg']['f1-score']
                })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    print(df.sort_values('accuracy', ascending=False).to_string(index=False))
    
    # Create bar plots for different metrics
    metrics = ['accuracy', 'avg_time_per_sample', 'throughput', 'f1_score']
    titles = ['Accuracy', 'Average Inference Time (s)', 'Throughput (samples/s)', 'F1 Score (macro avg)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Reshape data for grouped bar chart
        pivot_df = df.pivot(index='model', columns='activation', values=metric)
        
        # Plot
        pivot_df.plot(kind='bar', ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_xlabel('Model Architecture')
        axes[i].set_ylabel(title)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        if metric == 'avg_time_per_sample':
            # Lower is better for inference time
            axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "model_comparison.png"))
    # plt.show()  # Commented out to avoid blocking
    plt.close()
    
    # Save the best model(s) based on accuracy
    best_model_key = df.loc[df['accuracy'].idxmax()]
    print(f"\nBest model: {best_model_key['model']} with {best_model_key['activation']} activation")
    print(f"Accuracy: {best_model_key['accuracy']:.4f}")
    print(f"Inference time: {best_model_key['avg_time_per_sample']*1000:.2f} ms per sample")
    print(f"Throughput: {best_model_key['throughput']:.2f} samples/second")

if __name__ == "__main__":
    main()