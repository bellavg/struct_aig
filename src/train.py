# -*- coding: utf-8 -*-
from __future__ import print_function
import datetime
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io
from random import shuffle
from util import *
from dagnn import DAGNN  # We only need the DAGNN model wrapper
from src.constants import *
import copy

# --- Main training script for DAG Transformer ---

# 1. Argument Parser (Simplified)
# All model-specific choices are removed and hardcoded later.
parser = argparse.ArgumentParser(description='Train a DAG Transformer for graph representation learning.')
# --- Essential Settings ---
parser.add_argument('--data-name', default='final_structures6',
                    help='Graph dataset name (e.g., your AIG dataset file).')
parser.add_argument('--save-appendix', default='_dag_sat_encoder', help='Suffix for the save directory.')
parser.add_argument('--nvt', type=int, default=6, help='Number of different node types in your graphs.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size during training.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--hs', type=int, default=501, help='Hidden size of the model.')
parser.add_argument('--nz', type=int, default=56, help='Number of dimensions of latent vector z.')

# --- Optional Settings ---
parser.add_argument('--save-interval', type=int, default=10, help='How many epochs to wait before saving.')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='Reprocess data instead of using cached .pkl file.')
parser.add_argument('--continue-from', type=int, default=None, help="Checkpoint epoch to continue training from.")
parser.add_argument('--infer-batch-size', type=int, default=128, help='Batch size during inference/encoding.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--device', type=int, default=0, help='CUDA device index.')
parser.add_argument('--res_dir', type=str, default="results/", help='Directory to save results.')

args = parser.parse_args()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
print(args)

# 2. Prepare Data (Simplified)
# This section now only loads data in the required 'pyg' (PyTorch Geometric) format.
args.res_dir = os.path.join(args.res_dir, '{}{}'.format(args.data_name, args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
else:
    # Hardcoded to load ENAS-style data in 'pyg' format.
    train_data, test_data, graph_args = load_ENAS_graphs(
        args.data_name, n_types=args.nvt, fmt='pyg'
    )
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)

# Save command line input for reproducibility
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input saved.')

# 3. Prepare the Model (Simplified)
# This now directly instantiates the DAGNN wrapper with the best settings from the paper.
model = DAGNN(
    emb_dim=args.nvt + 2,
    hidden_dim=args.hs,
    out_dim=args.hs,
    max_n=graph_args.max_n,
    nvt=graph_args.num_vertex_type,
    START_TYPE=graph_args.START_TYPE,
    END_TYPE=graph_args.END_TYPE,
    hs=args.hs,
    nz=args.nz,
    num_nodes=args.nvt + 2,
    bidirectional=True,
    # --- Hardcoded Best Model Settings ---
    abs_pe='dagpe',  # Use depth-based positional encoding (DAGPE)
    dag_attention=1,  # Enable reachability-based attention (DAGRA)
    SAT=1,  # Enable Structure-Aware Transformer (SAT)
    gps=0  # Disable GPS model
)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
model.to(device)

if args.continue_from is not None:
    epoch = args.continue_from
    load_module_state(model, os.path.join(args.res_dir, f'model_checkpoint{epoch}.pth'))
    load_module_state(optimizer, os.path.join(args.res_dir, f'optimizer_checkpoint{epoch}.pth'))
    load_module_state(scheduler, os.path.join(args.res_dir, f'scheduler_checkpoint{epoch}.pth'))
    print(f"Loaded model state from epoch {epoch}")


# 4. Define Training and Inference Functions (Simplified)
def train(epoch):
    """ Main training loop for one epoch. """
    model.train()
    train_loss, recon_loss, kld_loss = 0, 0, 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    for i, (g, y) in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch_collated = model._collate_fn(g_batch)

            mu, logvar = model.encode(g_batch_collated)
            loss, recon, kld = model.loss(mu, logvar, g_batch_collated)

            pbar.set_description(f'Epoch: {epoch}, loss: {loss.item() / len(g_batch):.4f}, '
                                 f'recon: {recon.item() / len(g_batch):.4f}, '
                                 f'kld: {kld.item() / len(g_batch):.4f}')

            loss.backward()
            optimizer.step()

            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            g_batch = []

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_data):.4f}')
    return train_loss, recon_loss, kld_loss


def extract_latent(data):
    """ Encodes a dataset to get the latent representations (z vectors). """
    model.eval()
    Z, Y = [], []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data, desc="Extracting latents")):
        g_batch.append(copy.deepcopy(g))
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch_collated = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch_collated)
            Z.append(mu.cpu().detach().numpy())
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


def save_latent_representations(epoch):
    """ Extracts and saves the latent representations for train and test sets. """
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)

    latent_pkl_name = os.path.join(args.res_dir, f'{args.data_name}_latent_epoch{epoch}.pkl')
    latent_mat_name = os.path.join(args.res_dir, f'{args.data_name}_latent_epoch{epoch}.mat')

    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)

    scipy.io.savemat(latent_mat_name,
                     mdict={'Z_train': Z_train, 'Z_test': Z_test,
                            'Y_train': Y_train, 'Y_test': Y_test})


# 5. Main Training Execution
time_start = datetime.datetime.now()
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
if os.path.exists(loss_name):
    os.remove(loss_name)

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    train_loss, recon_loss, kld_loss = train(epoch)

    with open(loss_name, 'a') as loss_file:
        loss_file.write(f"{epoch} {train_loss / len(train_data):.4f} "
                        f"{recon_loss / len(train_data):.4f} {kld_loss / len(train_data):.4f}\n")

    scheduler.step(train_loss)

    if epoch % args.save_interval == 0 or epoch == 1:
        print(f"--- Saving model and latent representations at epoch {epoch} ---")
        torch.save(model.state_dict(), os.path.join(args.res_dir, f'model_checkpoint{epoch}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.res_dir, f'optimizer_checkpoint{epoch}.pth'))
        torch.save(scheduler.state_dict(), os.path.join(args.res_dir, f'scheduler_checkpoint{epoch}.pth'))
        save_latent_representations(epoch)

print("--- Final Latent Representation Extraction ---")
save_latent_representations(args.epochs)
print(f"TRAINING COMPLETE. Total time: {datetime.datetime.now() - time_start}")
