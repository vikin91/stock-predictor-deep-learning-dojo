import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTM

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(input_dim=2, 
             hidden_dim=model_info['hidden_dim'], 
             batch_size=model_info['batch_size'],
             output_dim=1, 
             num_layers=model_info['num_layers']).to(device)

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))


    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
       
    dummy_read = pd.read_csv(os.path.join(training_dir, 'train.csv'), header=None, names=None)
    num_rows_train = (len(dummy_read) // batch_size) * batch_size
    train_data = pd.read_csv(os.path.join(training_dir, 'train.csv'), header=None, names=None, nrows=num_rows_train)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
        
    # for dev
    print_every = 5
    batch_size=50
    clip=5 # gradient clipping
    criterion = loss_fn

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch
            
            # convert data into Tensors
            x_tensor = torch.Tensor(batch_X).unsqueeze(0)
            y_tensor = torch.Tensor(batch_y)
          
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)
            
            model.zero_grad()
            
            output = model(x_tensor)
           
            loss = criterion(output.squeeze(), y_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()   
            
            total_loss += loss.data.item()
        print("Epoch: {}, MSELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--hidden-dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--num-layers', type=int, default=4, metavar='N',
                        help='number of LSTM layers (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
                        help='learnign rate parameter (default: 0.0001)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTM(input_dim=2, 
                 hidden_dim=args.hidden_dim, 
                 batch_size=args.batch_size,
                 output_dim=1, 
                 num_layers=args.num_layers).to(device)

    print("Model loaded with hidden_dim {}, batch_size {}, num_layers {}.".format(
        args.hidden_dim, args.batch_size, args.num_layers
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'hidden_dim': args.hidden_dim,
            'batch_size': args.batch_size,
            'num_layers': args.num_layers,
        }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
