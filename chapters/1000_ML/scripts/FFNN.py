import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from b777_engine import plot_b777_engine

#run in a conda environment created with:
# conda create -n engine_pred python=3.9 numpy=1.23.5 matplotlib scikit-learn pytorch cpuonly -c pytorch -y conda activate engine_pred


class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, activation='relu'):
        super(FeedforwardNet, self).__init__()

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        act_module = activations.get(activation, nn.ReLU())

        layers = []
        prev_dim = input_dim
        # Create hidden layers
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            nn.init.xavier_uniform_(layers[-1].weight)  # Xavier init
            layers.append(act_module)
            prev_dim = h
        # Output layer
        out_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(out_layer.weight)
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_and_scale(input_file, output_file):
    X_raw = np.loadtxt(input_file).astype(np.float32)
    Y_raw = np.loadtxt(output_file).astype(np.float32)
    assert X_raw.shape[0] == Y_raw.shape[0], "Input and output data must have the same number of samples"

    # Scale features and targets
    scaler_X = StandardScaler().fit(X_raw)
    scaler_Y = StandardScaler().fit(Y_raw)
    X_scaled = scaler_X.transform(X_raw)
    Y_scaled = scaler_Y.transform(Y_raw)
    return X_scaled, Y_scaled, scaler_X, scaler_Y, X_raw, Y_raw


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig("../outputs/plot-losses.png", dpi=800, transparent=False)
    print("Saving figure to:", os.path.abspath("plot-losses.png"))
    plt.show()


def main(args):
    # Load and scale data
    X, Y, scaler_X, scaler_Y, X_raw, Y_raw = load_and_scale(args.input_file, args.output_file)
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)

    # Split into train/val/test
    total = len(dataset)
    test_size = int(total * args.test_split)
    val_size = int(total * args.val_split)
    train_size = total - test_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Define model
    model = FeedforwardNet(
        input_dim=X.shape[1], 
        hidden_units=args.hidden_units, 
        output_dim=Y.shape[1],
        activation=args.activation
        )
    model.to(args.device)
    print("Model architecture:")
    print(model)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_train_loss = running_loss / train_size
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        epoch_val_loss = val_loss / val_size
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Test evaluation (with inverse scaling)
    model.eval()
    preds_list = []
    y_true_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            preds_list.append(out.numpy())
            y_true_list.append(yb.numpy())
    preds_scaled = np.vstack(preds_list)
    y_true_scaled = np.vstack(y_true_list)
    # Inverse transform\    
    preds = scaler_Y.inverse_transform(preds_scaled)
    y_true = scaler_Y.inverse_transform(y_true_scaled)

    mse = np.mean((preds - y_true)**2)
    mae = np.mean(np.abs(preds - y_true))
    print(f"\nTest MSE (original scale): {mse:.6f}")
    print(f"Test MAE (original scale): {mae:.6f}")

    # Breakdown of error per output dimension
    per_dim_mae = np.mean(np.abs(preds - y_true), axis=0)
    print(f"MAE for thrust: {per_dim_mae[0]:.2f}, for SFC: {per_dim_mae[1]:.7f}")

    #plot
    plot_b777_engine(X_raw, Y_raw, model, scaler_X, scaler_Y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network for engine performance prediction.")
    parser.add_argument('--input_file', type=str, default='../data/b777_engine_inputs.dat', help='Path to input data file')
    parser.add_argument('--output_file', type=str, default='../data/b777_engine_outputs.dat', help='Path to output data file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[128, 128],
                        help='List of hidden layer sizes, e.g., --hidden_units 64 64')
    parser.add_argument('--activation', type=str,
                        choices=['relu','tanh','sigmoid','leaky_relu'], default='relu',
                        help='Activation function to use in hidden layers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of data for testing')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
