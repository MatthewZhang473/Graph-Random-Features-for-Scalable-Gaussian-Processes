import torch
import gpytorch
from tqdm.auto import tqdm


def train_model(
    model,
    likelihood,
    x_train,
    y_train,
    lr=0.01,
    max_iter=100,
    print_every=None,
    progress_desc="Training",
):
    model.train()
    likelihood.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []

    pbar = tqdm(total=max_iter, desc=progress_desc, leave=False)
    for step in range(max_iter):
        optimizer.zero_grad()
        train_output = model(x_train)
        train_loss = -mll(train_output, y_train)
        train_loss.backward()
        optimizer.step()

        loss_value = train_loss.item()
        losses.append(loss_value)

        if print_every and ((step + 1) % print_every == 0 or (step + 1) == max_iter):
            print(f"Iter {step + 1}/{max_iter} - Loss: {loss_value:.4f}")

        pbar.set_postfix({"loss": f"{loss_value:.4f}"})
        pbar.update(1)

    pbar.close()
    return losses


def evaluate_model(
    model,
    likelihood,
    x_train,
    y_train,
    x_test,
    y_test,
    orig_std,
):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        train_output = model(x_train)
        lml = mll(train_output, y_train) * y_train.size(0)

        pred = model.predict(x_test)
        mean = pred.mean
        rmse = orig_std * torch.sqrt(
            torch.mean((y_test.flatten() - mean.flatten()) ** 2)
        )
        nlpd = -pred.log_prob(y_test.flatten()) / y_test.numel()

    return lml.item(), rmse.item(), nlpd.item()
