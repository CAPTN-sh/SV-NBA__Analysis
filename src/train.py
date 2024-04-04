import numpy as np
import torch
import pickle
from pathlib import Path

def train_and_evaluate(model, train_dataloader, eval_dataloader, optimizer, criterion, model_save_path, pickle_save_path, model_name, epochs=500, mode='train', test_dataloader=None, patience=10):
    train_losses = []
    eval_losses = []
    # For returning values
    encoded_features = []
    all_inputs = []
    all_reconstructions = []

    model_save_path = Path(model_save_path)
    pickle_save_path = Path(pickle_save_path)

    model_dir = model_save_path / 'trained_models'
    model_dir.mkdir(parents=True, exist_ok=True)

    pickle_dir = pickle_save_path / Path(model_name).stem
    pickle_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'train':
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        
        total_iterations = len(train_dataloader)
        print_interval = total_iterations // 100  # Adjust this value as needed

        # Early stopping initialization
        early_stopping_counter = 0
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for i, (features, labels) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    features, labels = features.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                
                _, decoded = model(features)
                
                loss = criterion(decoded, labels)
                loss.backward()

                # Gradient clipping before optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                # if i % print_interval == 0:
                #     print(f"Epoch [{epoch+1}/{epochs}], Iteration [{i+1}/{total_iterations}], Loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / len(train_dataloader)
            train_losses.append(epoch_loss)

            # Evaluate model to get eval_loss and possibly the last epoch's data
            eval_loss, encoded_feats, inputs, reconstructions = evaluate_model(model, eval_dataloader, criterion)
            eval_losses.append(eval_loss)
            scheduler.step(eval_loss)
            current_lr = scheduler.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}, Current Learning Rate: {current_lr}")

            # Early stopping logic
            if eval_loss < best_loss:
                best_loss = eval_loss
                early_stopping_counter = 0
                best_epoch = epoch  # 

                # Saving data when model is the best
                encoded_features_best = encoded_feats
                all_inputs_best = inputs
                all_reconstructions_best = reconstructions

                # Save the best model
                torch.save(model.state_dict(), model_dir / f"{model_name}_best.pth")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break
            
        # Load the best model for final evaluation or use
        model.load_state_dict(torch.load(model_dir / f"{model_name}_best.pth"))

        save_data_to_pickle(pickle_dir, "encoded_features.pkl", encoded_features_best)
        save_data_to_pickle(pickle_dir, "all_inputs.pkl", all_inputs_best)
        save_data_to_pickle(pickle_dir, "all_reconstructions.pkl", all_reconstructions_best)

    elif mode == 'test':
        # Determine which DataLoader to use for testing
        test_loader = test_dataloader if test_dataloader is not None else eval_dataloader
        
        # Load the model's state dictionary
        model_load_path = model_dir / f"{model_name}_best.pth"  
        model.load_state_dict(torch.load(model_load_path))
        model.eval()

        # Perform evaluation on the chosen test set
        eval_loss, encoded_features, all_inputs, all_reconstructions = evaluate_model(model, test_loader, criterion)
        eval_losses.append(eval_loss)
        
        # Save encoded_features, all_inputs, all_reconstructions to pickle
        save_data_to_pickle(pickle_dir, "encoded_features_train.pkl", encoded_features)
        save_data_to_pickle(pickle_dir, "all_inputs_train.pkl", all_inputs)
        save_data_to_pickle(pickle_dir, "all_reconstructions_train.pkl", all_reconstructions)

        print(f"Test Mode, Eval Loss: {eval_loss:.4f}")

    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    return train_losses, eval_losses, encoded_features, all_inputs, all_reconstructions

def save_data_to_pickle(save_path, file_name, data):
    with open(save_path / file_name, "wb") as f:
        pickle.dump(data, f)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    eval_loss = 0.0
    all_encoded = []
    all_inputs = []
    all_reconstructions = []
    with torch.no_grad():
        for features, labels in dataloader:
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            
            encoded, decoded = model(features)
            loss = criterion(decoded, labels)
            eval_loss += loss.item()

            all_encoded.append(encoded.cpu().numpy())
            all_inputs.append(features.cpu().numpy())
            all_reconstructions.append(decoded.cpu().numpy())
            
    eval_loss /= len(dataloader)

    encoded_features = np.concatenate(all_encoded, axis=0) if all_encoded else []
    all_inputs = np.concatenate(all_inputs, axis=0) if all_inputs else []
    all_reconstructions = np.concatenate(all_reconstructions, axis=0) if all_reconstructions else []

    return eval_loss, encoded_features, all_inputs, all_reconstructions