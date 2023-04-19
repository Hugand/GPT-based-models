import torch
from torch import nn
import numpy as np
from modules.TransformerBlock import TransformerBlock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ClassificationHead(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, output_size, dropout=0.1):
        super().__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Linear(in_features=n_embeddings * embedding_dim, out_features=embedding_dim),
            nn.Dropout(p=dropout),
            nn.GELU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=output_size),
            nn.Dropout(p=dropout),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        return x

class NanoGPTClassifier(nn.Module):
    def __init__(self, output_size, n_transformer_blocks, n_embeddings, embedding_dim):
        super().__init__()
        self.n_transformer_blocks = n_transformer_blocks
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.output_size = output_size

        # Layers
        self.embedding = nn.Embedding(n_embeddings, embedding_dim).to(device)
        self.transformer_blocks = [TransformerBlock(10, embedding_dim, False) for _ in range(n_transformer_blocks)]
        self.output_head = ClassificationHead(n_embeddings, embedding_dim, output_size).to(device)

        # Initialize weights
        self._init_weights(self.embedding)
        for block in self.transformer_blocks:
            self._init_weights(block)
        self._init_weights(self.output_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, features):
        # Embedding
        X = self.embedding(features.to(device)).to(device)
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            X = transformer_block(X)

        flattened = X.view(X.size(0), -1)
        # Classifier layers
        X = self.output_head(flattened)
        
        return X

    def fit(self,
            X, y,
            optimizer, loss_criterion,
            epochs=10,
            batch_size=64,
            save_frequency=10,
            max_linear_scheduler_epochs=2000
    ):
        losses = []
        loss = 0
        batch_progress = 0
        n_batches = np.round(len(X) / batch_size).astype(np.int)
        X_size = len(X)
        print(X.shape)
        
        X = torch.reshape(X, (n_batches, batch_size, X.shape[1])).to(device)
        # X = torch.from_numpy(X).to(device)

        y = torch.reshape(y, (n_batches, batch_size, y.shape[1])) \
            .type(torch.FloatTensor).to(device)
        y = torch.argmax(y, dim=2)
        # y = torch.from_numpy(y).to(device)

        print(X.shape, y.shape, n_batches)

        # The -1 is to prevent division by 0
        max_annealing_scheduler_epochs = n_batches * epochs - max_linear_scheduler_epochs - 1

        linear_scheduler = LinearLR(optimizer, start_factor=1/max_linear_scheduler_epochs, end_factor=1.0, total_iters=max_linear_scheduler_epochs)
        annealing_scheduler = CosineAnnealingLR(optimizer, T_max=max_annealing_scheduler_epochs, eta_min=0.0)
        learning_rates = []
        train_accuracies = []

        print("Starting training...")
        for epoch in range(epochs):
            loss = 0
            train_acc = 0
            print(f'Epoch {epoch}/{epochs} - ', end="")
            
            for i in range(n_batches):

                # Generate batch noisy images
                optimizer.zero_grad()
                
                # compute reconstructions
                outputs = self.forward(X[i])

                # compute training reconstruction loss
                train_loss = loss_criterion(outputs, y[i])

                enc_outputs = torch.argmax(outputs, dim=1)

                train_acc += torch.sum(enc_outputs == y[i]) / len(y[i])

                # compute accumulated gradients for generator and discriminator
                train_loss.backward()
                
                # perform parameter update based on current gradients only for the generator
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

                # Update learning rate with the schedulers
                if epochs * i < max_linear_scheduler_epochs:
                    linear_scheduler.step()
                else:
                    annealing_scheduler.step()

                #progress += step_size
                batch_progress += 1
                print('#', end="")

                
            
            after_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(after_lr)
            losses.append(loss)

            train_acc = train_acc / n_batches
            train_accuracies.append(train_acc)

            print(f', loss: {loss}, acc: {train_acc}')

            if epoch % save_frequency == 0:
                torch.save(self.state_dict(), 'nano-gpt-classifier.model')

        torch.save(self.state_dict(), 'nano-gpt-classifier-final.model')
            
        return losses, train_accuracies, learning_rates