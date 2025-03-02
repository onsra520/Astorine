import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
from nlp.nlp_utils import *
from nlp.chatbot import CustomNN, DialogueDataset

class chatbottraining:
    def __init__(
        self,         
        path: str,
        training_data: dict,      
        hidden_size: int = 64,
        batch_size: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 3e-4,
        num_epochs: int = 200,
        validation_split: float = 0.2, 
        patience: int = 10,
        use_cuda: bool = True
    ) -> None:
        
        self.intents_data = training_data
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs    
        self.validation_split = validation_split
        self.patience = patience
        self.use_cuda = use_cuda and torch.cuda.is_available()
        X_train, Y_train = self._prepare_dataset()
        self.input_size = len(X_train[0])
        self.output_size = len(self.tags)
        dataset_size = len(X_train)
        indices = list(range(dataset_size))
        val_split = int(np.floor(self.validation_split * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[val_split:], indices[:val_split]
        X_train_split = [X_train[i] for i in train_indices]
        Y_train_split = [Y_train[i] for i in train_indices]
        X_val = [X_train[i] for i in val_indices]
        Y_val = [Y_train[i] for i in val_indices]
        
        train_dataset = DialogueDataset(X_train_split, Y_train_split)
        val_dataset = DialogueDataset(X_val, Y_val)
        num_workers = 0 

        self.train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.use_cuda
        )
        
        self.val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.use_cuda
        )
        
        self._training(path)

    def _prepare_dataset(self):
        self.all_words, self.tags, xy = [], [], []
        ignore_words = ['?', '!', '.', ',']
        
        patterns_list = []
        tags_list = []

        total_steps = 100
        with tqdm(total=total_steps, desc="Preparing the training dataset") as main_bar:
            with tqdm(total=len(self.intents_data['intents']), desc="Tokenizing patterns in parallel", leave=False) as sub_bar1:
                for intent in self.intents_data['intents']:
                    if "tag" in intent and "patterns" in intent:
                        tag = intent['tag']
                        self.tags.append(tag)
                        for pattern in intent['patterns']:
                            patterns_list.append(pattern.lower())
                            tags_list.append(tag)
                        
                        all_tokenized_patterns = tokenize_parallel(patterns_list)
                        for i, tokenized_pattern in enumerate(all_tokenized_patterns):
                            self.all_words.extend(tokenized_pattern)
                            xy.append((tokenized_pattern, tags_list[i]))
                        sub_bar1.update(1)
                main_bar.update(25) 
                    
            with tqdm(total=100, desc="Processing word stems", leave=False) as sub_bar2:           
                filtered_words = [word for word in self.all_words if word not in ignore_words]
                sub_bar2.update(25)
                
                stemmed_words = batch_process_stems(filtered_words)
                sub_bar2.update(50)
                
                self.all_words = sorted(set(stemmed_words))
                sub_bar2.update(25)
                main_bar.update(25)

            with tqdm(total=len(xy), desc="Augmenting dataset", leave=False) as sub_bar3:
                augmented_xy = []
                for pattern_words, tag in xy:
                    augmented_xy.append((pattern_words, tag))
                    if len(pattern_words) > 3: 
                        dropout_words = pattern_words.copy()
                        drop_idx = np.random.randint(0, len(pattern_words))
                        dropout_words.pop(drop_idx)
                        augmented_xy.append((dropout_words, tag))
                    if len(pattern_words) > 3: 
                        shuffled_words = pattern_words.copy()
                        if len(shuffled_words) > 1:
                            i, j = np.random.choice(range(len(shuffled_words)), 2, replace=False)
                            shuffled_words[i], shuffled_words[j] = shuffled_words[j], shuffled_words[i]
                        augmented_xy.append((shuffled_words, tag))
                    sub_bar3.update(1)
                main_bar.update(25)
                    
            with tqdm(total=len(xy), desc="Creating bag-of-words vectors", leave=False) as sub_bar4:
                xy = augmented_xy
                X_train = []
                Y_train = []
                batch_size = 100 
                for i in range(0, len(xy), batch_size):
                    batch_xy = xy[i:i+batch_size]
                    batch_sentences = [pattern for pattern, _ in batch_xy]
                    batch_tags = [tag for _, tag in batch_xy]
                    batch_bow = create_bow_matrix(batch_sentences, self.all_words)
                    for j, tag in enumerate(batch_tags):
                        X_train.append(batch_bow[j])
                        label = self.tags.index(tag)
                        Y_train.append(label)
                    sub_bar4.update(len(batch_xy))
                main_bar.update(25)
        return X_train, Y_train

    def _training(self, path: str) -> None:
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        model = CustomNN(
            self.input_size, 
            self.hidden_size, 
            self.output_size, 
            self.dropout_rate
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4  
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        device_info = {}
        if self.use_cuda:
            current_device = torch.cuda.current_device()
            device_info['Device'] = f"GPU: {torch.cuda.get_device_name(current_device)}"
            device_info['VRAM'] = f"{torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB"
        else:
            device_info['Device'] = "CPU"

        with tqdm(total=self.num_epochs, desc="Training") as epoch_bar:
            try:
                for epoch in range(self.num_epochs):
                    model.train()
                    epoch_loss = 0.0
                    batch_count = 0
                    correct = 0
                    total = 0
                    train_loader_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False)
                    
                    for (words, labels) in train_loader_iter:
                        words = words.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.long)
                        
                        if self.use_cuda:
                            with torch.amp.autocast('cuda'):
                                outputs = model(words)
                                loss = criterion(outputs, labels)
                        
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(words)
                            loss = criterion(outputs, labels)
                            
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        epoch_loss += loss.item()
                        batch_count += 1
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        train_acc = correct / total if total > 0 else 0
                        train_loader_iter.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'acc': f"{train_acc:.4f}"
                        })
                    
                    train_loss = epoch_loss / batch_count if batch_count > 0 else 0
                    train_acc = correct / total if total > 0 else 0
                    model.eval()
                    val_loss = 0.0
                    val_batch_count = 0
                    val_correct = 0
                    val_total = 0
                    
                    val_loader_iter = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]", leave=False)
                    
                    with torch.no_grad():
                        for (words, labels) in val_loader_iter:
                            words = words.to(device, dtype=torch.float) 
                            labels = labels.to(device, dtype=torch.long)
                            
                            if self.use_cuda:
                                with torch.amp.autocast('cuda'):
                                    outputs = model(words)
                                    loss = criterion(outputs, labels)
                            else:
                                outputs = model(words)
                                loss = criterion(outputs, labels)
                            
                            val_loss += loss.item()
                            val_batch_count += 1
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                            val_acc = val_correct / val_total if val_total > 0 else 0
                            val_loader_iter.set_postfix({
                                'loss': f"{loss.item():.4f}",
                                'acc': f"{val_acc:.4f}"
                            })
                    
                    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
                    val_acc = val_correct / val_total if val_total > 0 else 0
                    scheduler.step(avg_val_loss)

                    epoch_bar.update(1)
                    epoch_bar.set_description(f"Epoch {epoch+1}/{self.num_epochs}")
                    epoch_bar.set_postfix({
                        **device_info,
                        'Train Loss': f"{train_loss:.4f}",
                        'Train Acc': f"{train_acc:.4f}",
                        'Val Loss': f"{avg_val_loss:.4f}",
                        'Val Acc': f"{val_acc:.4f}"
                    })

                    if early_stop_counter >= self.patience:                              
                        print(f"Early stopping at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
                        break

                    if train_loss < 0.01 and avg_val_loss < 0.1:
                        print(f"Reached desired performance at epoch {epoch+1}")
                        break
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        early_stop_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        early_stop_counter += 1
                
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    
                print(f'Final validation loss: {best_val_loss:.4f}')
                checkpoint = {
                    "model_state": model.state_dict(),
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "output_size": self.output_size,
                    "all_words": self.all_words,
                    "tags": self.tags
                }
                torch.save(checkpoint, path)
                print(f"Model saved to {path}")
            
            except RuntimeError as e:
                print(f"RuntimeError occurred: {str(e)}")
                print(f"Current device: {device}")
                if self.use_cuda:
                    print(f"CUDA available: {torch.cuda.is_available()}")
                    print(f"Current CUDA device: {torch.cuda.current_device()}")
                if self.use_cuda:
                    print("Attempting to fall back to CPU...")
                    self.use_cuda = False
                    self._training(path)
                else:
                    raise