import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import wandb

# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from DCN import DCN


def evaluate(model, test_loader):
    # y_test = []
    y_pred = []
    # for data, target in test_loader:
    for data in test_loader:
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        # y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))

    # y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    # return (normalized_mutual_info_score(y_test, y_pred), adjusted_rand_score(y_test, y_pred))
    return y_pred


def solver(args, model, train_loader, test_loader, run=None):
    _rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    # pred_list = []
    # nmi_list = []
    # ari_list = []

    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader, run=run)

        # model.eval()
        # NMI, ARI = evaluate(model, test_loader)  # evaluation on test_loader
        # nmi_list.append(NMI)
        # ari_list.append(ARI)

        # print("\nEpoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}\n".format(e, NMI, ARI))
        print(f"\nEpoch: {e:02d}")

    preds = evaluate(model, test_loader)

    # return _rec_loss_list, nmi_list, ari_list
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Clustering Network")

    # Dataset parameters
    parser.add_argument("--dir", default="../Dataset/mnist", help="dataset directory")
    parser.add_argument("--input-dim", type=int, default=28 * 28, help="input dimension")
    parser.add_argument("--n-classes", type=int, default=10, help="output dimension")

    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (default: 5e-4)")
    parser.add_argument("--batch-size", type=int, default=256, help="input batch size for training")
    parser.add_argument("--epoch", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--pre-epoch", type=int, default=50, help="number of pre-train epochs")
    parser.add_argument("--pretrain", action="store_true", help="whether use pre-training")

    # Model parameters
    parser.add_argument("--lamda", type=float, default=1, help="coefficient of the reconstruction loss")
    parser.add_argument(
        "--beta", type=float, default=1, help=("coefficient of the regularization term on " "clustering")
    )
    parser.add_argument("--hidden-dims", nargs="+", default=[500, 500, 2000], help="learning rate (default: 1e-4)")
    parser.add_argument("--latent-dim", type=int, default=10, help="latent space dimension")
    parser.add_argument("--n-clusters", type=int, default=10, help="number of clusters in the latent space")

    # Utility parameters
    parser.add_argument("--n-jobs", type=int, default=1, help="number of jobs to run in parallel")
    parser.add_argument("--cuda", action="store_true", help="whether to use GPU")
    parser.add_argument("--log-interval", type=int, default=100, help=("how many batches to wait before logging"))

    # Results parameters
    parser.add_argument(
        "--save-dir", type=str, default=f"./results/{datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')}"
    )

    args = parser.parse_args()

    # setup
    os.makedirs(args.save_dir, exist_ok=False)
    run = wandb.init("rshwndsz/themis-dcn")

    # Load data
    train_data = torch.load(os.path.join(args.dir, "train_data.pt"), weights_only=True).to(torch.float)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    test_data = torch.load(os.path.join(args.dir, "train_data.pt"), weights_only=True).to(torch.float)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # Main body
    model = DCN(args)
    # rec_loss_list, nmi_list, ari_list = solver(args, model, train_loader, test_loader)
    predictions = solver(args, model, train_loader, test_loader, run=run)

    # Save predictions
    df = pd.DataFrame(predictions)
    df.to_csv(
        os.path.join(
            args.save_dir,
            f"dcn_predictions--{args.n_clusters}-clusters--{'-'.join([str(x) for x in args.hidden_dims])}.csv",
        )
    )
