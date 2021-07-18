"""Training procedure for neural operator network."""

from datetime import datetime
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from tqdm import tqdm

def train_operator(model,
                   lp_loss,
                   device,
                   optimizer,
                   scheduler,
                   epochs,
                   train_loader,
                   test_loader,
                   PATH=None,
                   y_normalizer=None):
    """Training loop."""

    output_freq = 5
    checkpoint_freq = 250

    if PATH is None:
        PATH = Path.cwd()

    results_dir = PATH.mkdir('results', exists_ok=True)
    model_dir = PATH.mkdir('model', exists_ok=True)
    log_dir = PATH.mkdir('log', exists_ok=True)
    checkpoint = log_dir+"log/{}.pt".format(model.name())

    train_log = []
    tr_err = []
    tst_err = []

    start_time = default_timer()

    for ep in tqdm(range(epochs)):

        model.train()
        t1 = default_timer()
        train_mse = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            if y_normalizer is not None:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

            b = x.shape[0]
            loss = lp_loss(out.view(b, -1), y.view(b, -1))
            loss.backward()

            optimizer.step()
            train_mse += loss.item()

        scheduler.step()

        model.eval()
        abs_err = 0.0
        rel_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)

                if y_normalizer is not None:
                    out = y_normalizer.decode(model(x))

                b = x.shape[0]
                abs_err += lp_loss.abs(out.view(b, -1),
                                       y.view(b, -1)).item()
                rel_err += lp_loss(out.view(b, -1),
                                   y.view(b, -1)).item()

        train_mse /= len(train_loader)
        abs_err /= len(test_loader)
        rel_err /= len(test_loader)

        tr_err.append(train_mse)
        tst_err.append(rel_err)

        t2 = default_timer()

        epoch_vals = ("epoch {} \n"
                      "values:\n"
                      "elapsed time: {:.3f}\n"
                      "training mse: {:.5f}\n"
                      "absolute error: {:5f}\n"
                      "relative error: {:.5f}\n\n").format(
                          ep, t2 - t1, train_mse, abs_err, rel_err)
        train_log.append(epoch_vals)

        if ep % output_freq == 0:
            print(epoch_vals)

        if ep % checkpoint_freq == 0:
            torch.save(
                {
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'log': train_log,
                }, checkpoint)

    end_time = default_timer()

    final_vals = ("final values:\n"
                  "elapsed time: {:.3f}\n"
                  "training mse: {:.5f}\n"
                  "relative error: {:.5f}\n\n").format(end_time - start_time,
                                                       train_mse, rel_err)
    print(final_vals)
    train_log.append(final_vals)

    log = log_dir.joinpath('log_{}.txt'.format(model.name()))
    fd = open(log, "a")
    for string in train_log:
        fd.write(string)
    fd.close()

    torch.save(model, model_dir)
    np.savetxt(results_dir, tr_err)
    np.savetxt(results_dir, tst_err)
