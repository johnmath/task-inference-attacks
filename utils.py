from tqdm import tqdm
import torch
import random
import numpy as np
from scipy import interpolate
from sklearn.metrics import roc_curve


def train_mtl_vision_model(
        multitask_model, 
        dataloader, 
        epochs, 
        criterion, 
        optimizer, 
        eval_loader=None, 
        scheduler=None, 
        device="cuda", 
        multilabel=False, 
        eval_epochs=None, 
        eval_at_end=False,
        ood_eval=False,
        warm_start_epochs=None,
        accumulation_steps=1,
    ):
    """"Do Warmup by using pretrained model to locally train final layers"""
    # Warm start setup
    ####### 
    multitask_model.train()

    # Train only linear layers
    if warm_start_epochs:
        for p in multitask_model.shared_layers[0].parameters():
            p.requires_grad_(False)


    loop = tqdm(range(epochs), desc="Training...", unit="epoch", colour="green", position=0, leave=True)
    multitask_model = multitask_model.to(device)
    
    # Clear gradients at the beginning
    optimizer.zero_grad()
    eval_str = ""  # Reset for each epoch
    for epoch in loop:

        running_loss = 0.0
        total_tasks = 0
        overall_accuracy = 0.0
        steps_since_update = 0
        
        for i, (data, labels, task_id) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)
            task_id = task_id.to(device)

            # Handle warm start
            if warm_start_epochs and (warm_start_epochs > epoch):
                pass
            elif warm_start_epochs and (warm_start_epochs <= epoch):
                with torch.no_grad():
                    print(f"[Debug] Turning off warm start before epoch {epoch + 1}")
                    for p in multitask_model.shared_layers[0].parameters():
                        p.requires_grad_(True)
                    warm_start_epochs = None

            # Process tasks within the batch
            multitask_loss = 0.0
            for task in range(data.shape[0]):
                total_tasks += 1
                outputs = multitask_model(data[task], task_id[task][0])
                loss = criterion(outputs, labels[task])
                
                # Scale by batch size only (accumulation will be handled later)

                scaled_loss = loss / data.shape[0]
                multitask_loss += scaled_loss
                running_loss += loss.item()
                
                if not multilabel:
                    overall_accuracy += (outputs.argmax(dim=1) == labels[task]).float().mean()
            
            # Call backward once with the accumulated task losses
            multitask_loss.backward()
            steps_since_update += 1
            
            # Update only after accumulation_steps
            if steps_since_update >= accumulation_steps:
                normalize_gradients(multitask_model.shared_layers.parameters(),)
                torch.nn.utils.clip_grad_norm_(multitask_model.task_specific_layers.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()
                steps_since_update = 0
        
        # Make sure to update at the end of epoch if there are any remaining gradients
        if steps_since_update > 0:
            # torch.nn.utils.clip_grad_norm_(multitask_model.shared_layers.parameters(), max_norm=2.0)
            normalize_gradients(multitask_model.shared_layers.parameters(),)
            torch.nn.utils.clip_grad_norm_(multitask_model.task_specific_layers.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Evaluation code is fine
        if eval_epochs and (epoch % eval_epochs)==0 and eval_loader is not None:
            outs = eval_mtl_vision_model(
                multitask_model, 
                eval_loader, 
                criterion, 
                device=device, 
                multilabel=multilabel,
                sampled_final_layers=ood_eval,
            )
            multitask_model.train()

            if not multilabel:
                eval_str = f" || test loss={outs[0]:.3f}; test acc={outs[1]:.3f}"
            else:
                eval_str = f" || test loss={outs:.3f}"

        # Display metrics
        if not multilabel:
            loop.set_postfix_str(f"loss={running_loss/total_tasks:.4f}; acc={overall_accuracy/total_tasks:.3f}" + eval_str)
        else:
            loop.set_postfix_str(f"loss={running_loss/total_tasks:.4f}" + eval_str)

        if scheduler:
            scheduler.step()
    
    multitask_model.eval()
    
    if eval_at_end and eval_loader is not None:
        outs = eval_mtl_vision_model(
            multitask_model, 
            eval_loader, 
            criterion, 
            device=device, 
            multilabel=multilabel,
            sampled_final_layers=ood_eval
        )
        return multitask_model, outs
    else:
        return multitask_model


def eval_mtl_vision_model(multitask_model, 
        dataloader, 
        criterion, 
        device="cuda", 
        multilabel=False, 
        sampled_final_layers=None,
    ):

    multitask_model.eval()
    multitask_model = multitask_model.to(device)
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_tasks = 0
    
    with torch.no_grad():
        for i, (data, labels, task_id) in tqdm(enumerate(dataloader), desc="Testing..."):
            data = data.to(device)
            labels = labels.to(device)
            
            for task in range(data.shape[0]):
                total_tasks += 1
                
                if not sampled_final_layers:
                    # Standard evaluation: use the correct task head
                    outputs = multitask_model(data[task], task_id[task][0])
                    loss = criterion(outputs, labels[task])
                    total_loss += loss.item()
                    
                    if not multilabel:
                        accuracy = (outputs.argmax(dim=1) == labels[task]).float().mean()
                        total_accuracy += accuracy.item()
                else:
                    # Zero-shot evaluation: sample random task heads
                    task_losses = []
                    task_accuracies = []
                    
                    # Sample random task IDs
                    random_task_ids = random.sample(range(len(multitask_model.task_specific_layers)), 
                                                  min(sampled_final_layers, len(multitask_model.task_specific_layers)))
                    
                    for random_task_id in random_task_ids:
                        outputs = multitask_model(data[task], random_task_id)
                        task_loss = criterion(outputs, labels[task]).item()
                        task_losses.append(task_loss)
                        
                        if not multilabel:
                            task_accuracy = (outputs.argmax(dim=1) == labels[task]).float().mean().item()
                            task_accuracies.append(task_accuracy)
                    
                    # Average over the sampled task heads
                    total_loss += sum(task_losses) / len(task_losses)
                    
                    if not multilabel:
                        total_accuracy += sum(task_accuracies) / len(task_accuracies)
    
    # Calculate averages across all tasks
    avg_loss = total_loss / total_tasks if total_tasks > 0 else float('inf')
    
    if multilabel:
        return avg_loss
    else:
        avg_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
        return avg_loss, avg_accuracy


def normalize_gradients(parameters):
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.norm() + 1e-8  # Add small epsilon for stability
            param.grad.data.div_(param_norm)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def quantile_stats(quantiles, test_statistic, true_labels, tpr=None, fpr=None, thresholds=None,):

    if not tpr or not fpr or not thresholds:
        fpr, tpr, thresholds = roc_curve(true_labels, test_statistic)
    
    all_results = {"quantile": quantiles, "tpr": [], "fpr": [], "acc": []}
    for quantile in quantiles:
        q_thresh = torch.quantile(test_statistic, quantile).item()

        tpr_intrp = interpolate.interp1d(thresholds, tpr)
        fpr_intrp = interpolate.interp1d(thresholds, fpr)

        tpr_at_q = tpr_intrp(q_thresh).item()
        fpr_at_q = fpr_intrp(q_thresh).item()
        acc = (tpr_at_q + (1-fpr_at_q))/2

        all_results["tpr"].append(tpr_at_q)
        all_results["fpr"].append(fpr_at_q)
        all_results["acc"].append(acc)
 
    return all_results