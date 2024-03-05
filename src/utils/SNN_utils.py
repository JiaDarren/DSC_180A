import numpy as np
import torch
from tqdm import tqdm
from utils.feature_eval_utils import get_metrics

def train_SNN(
        model, 
        loss_fn, 
        optimizer, 
        X_train_tensor, 
        y_train_tensor, 
        X_valid_tensor, 
        y_valid_tensor, 
        n_epochs=300,
        batch_size=16, 
        display=True,
        file_name="nn_base_params",
        MODEL_PATH='../models'
        ):
    
    batch_start = torch.arange(0, X_train_tensor.shape[0], batch_size)
    train_history = []
    valid_history = []
    best_valid = np.inf
    
    # training loop
    if display: bar = tqdm(range(n_epochs))
    else: bar = range(n_epochs)
    for epoch in bar:
        model.train()
        if display: bar.set_description(f"Epoch {epoch+1}")
        for start in batch_start:
            # take a batch
            X_batch = X_train_tensor[start:start+batch_size]
            y_batch = y_train_tensor[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred_train = model(X_train_tensor)
        y_pred_valid = model(X_valid_tensor)
        train_loss = float(loss_fn(y_pred_train, y_train_tensor))
        valid_loss = float(loss_fn(y_pred_valid, y_valid_tensor))
        
        if display: 
            bar.set_postfix(
                Train_MSE = float(train_loss), 
                Valid_MSE = float(valid_loss)
            )
        train_history.append(train_loss)
        valid_history.append(valid_loss)

        if (valid_loss < best_valid):
            torch.save(model.state_dict(), f'{MODEL_PATH}/{file_name}.pt')

    if display:
        print(f"Best Epoch: {np.argmin(valid_history)}")
        print(f"Loss: {np.min(valid_history)}")
    model.load_state_dict(torch.load(f'{MODEL_PATH}/{file_name}.pt'))
    return model, train_history, valid_history

def find_best_prob(pred_prob, actual, display=True):
    best_prob = 0
    best_ratio = np.inf
    prob_interval = 0.05
    with np.errstate(divide='ignore'):
        if display:
            print(
                f"i\t" + 
                f"ratio\t" + 
                f"roc_auc\t" + 
                f"acc\t" + 
                f"conf_matrix" + 
                "\n==================================================="
            )
        for i in np.arange(0, 1.0, prob_interval):
            pred = pred_prob > i
            metrics = get_metrics(pred, actual, display=False)
            

            TPR = metrics[4]
            FPR = metrics[5]
            PLR = TPR/FPR
            NLR = FPR/TPR
            roc_auc = metrics[-1]
            ratio = (PLR + NLR)/2

            if display:
                print(
                    f"{i:.2f}\t" + 
                    f"{ratio:0.3f}\t" + 
                    f"{roc_auc:0.3f}\t" + 
                    f"{((metrics[0] + metrics[3]) / np.sum(metrics[:4]) * 100):.2f}\t" + 
                    f"{metrics[:4]}"
                )

            if (ratio < best_ratio):
                best_ratio = ratio
                best_prob = i
    if display: print(f"Best Threshold: {best_prob:0.2f}")
    return best_prob