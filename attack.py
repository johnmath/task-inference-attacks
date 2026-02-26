import torch
from tqdm import tqdm
from pathlib import Path
import random


class TaskInference:
    
    def __init__(self, multitask_model, in_dataset, out_dataset,):

        self.model = multitask_model
        self.in_data = in_dataset
        self.out_data = out_dataset
        self.embeddings = None

    def generate_all_embeddings(
            self, 
            save=None, 
            device="cuda:0", 
            augs=None, 
            aug_transform=None, 
            pos_only=False, 
            whitening=True, 
            sample=None,
        ):
        data_type="vision"

        self.model.eval()
        self.model = self.model.to(device)
        
        in_data_loop = tqdm(range(len(self.in_data)), desc="Generating IN Embeddings...", unit="task", colour="green", position=0, leave=True)
        
        if sample:
            self.in_data.enable_subsampling(sample)
        else:
            self.in_data.disable_subsampling()
        in_embeddings = []
        in_true_labels = []

        if pos_only: 
            raise Exception("Dataset does not support positive sampling")
        else:
            pass

        # Generate IN embeddings
        with torch.no_grad():
            for i in in_data_loop:
                if data_type == "vision":
                    task_data, task_labels, _ = self.in_data[i]

                    if augs:
                        all_task_data = []
                        for im in task_data:
                            for _ in range(augs):
                                all_task_data.append(aug_transform(im))
                        all_task_data = torch.stack(all_task_data)
                        task_data = all_task_data.clone()

                    task_data = task_data.to(device)
                    embs = self.model.forward_emb(task_data).cpu()
                    if augs:
                        raise Exception
                        embs = embs.reshape(len(self.in_data[i][0]), augs, -1)
                        embs = embs.mean(dim=1)
                    
                in_true_labels.append(task_labels)
                in_embeddings.append(embs)

        out_data_loop = tqdm(range(len(self.out_data)), desc="Generating OUT Embeddings...", unit="task", colour="green", position=0, leave=True)
        
        if sample:
            self.out_data.enable_subsampling(sample)
        else:
            self.out_data.disable_subsampling()
        
        out_embeddings = []
        out_true_labels = []

        # Generate OUT embeddings
        with torch.no_grad():
            for i in out_data_loop:
                if data_type == "vision":
                    task_data, task_labels, _ = self.out_data[i]

                    if augs:
                        all_task_data = []
                        for im in task_data:
                            for _ in range(augs):
                                all_task_data.append(aug_transform(im))
                        all_task_data = torch.stack(all_task_data)
                        task_data = all_task_data.clone()

                    task_data = task_data.to(device)
                    embs = self.model.forward_emb(task_data).cpu()
                    if augs:
                        embs = embs.reshape(len(self.out_data[i][0]), augs, -1)
                        embs = embs.mean(dim=1)
                    
                out_true_labels.append(task_labels)
                out_embeddings.append(embs)
            
        self.in_embeddings = in_embeddings
        self.out_embeddings = out_embeddings

        if whitening:
            self.whitening_transform_mean_in = []
            self.whitening_transform_mean_out = []
            self.whitening_transform_cov_in = []
            self.whitening_transform_cov_out = []

            for i in range(len(self.in_embeddings)):
                mu, cov = self._compute_whitening_transform(i, in_data=True)
                self.whitening_transform_mean_in.append(mu)
                self.whitening_transform_cov_in.append(cov)
            
            for j in range(len(self.out_embeddings)):
                mu, cov = self._compute_whitening_transform(j, in_data=False)
                self.whitening_transform_mean_out.append(mu)
                self.whitening_transform_cov_out.append(cov)

            self.whitening_transform_mean_in = torch.stack(self.whitening_transform_mean_in)
            self.whitening_transform_mean_out = torch.stack(self.whitening_transform_mean_out)
            self.whitening_transform_cov_in = torch.stack(self.whitening_transform_cov_in)
            self.whitening_transform_cov_out = torch.stack(self.whitening_transform_cov_out)
        

        self.in_labels = in_true_labels
        self.out_labels = out_true_labels

        if save:
            Path(f"{save}").mkdir(parents=True, exist_ok=True)
            torch.save(self.in_embeddings, f"{save}/in_embeddings.pth")
            torch.save(self.out_embeddings, f"{save}/out_embeddings.pth")
            torch.save(self.in_labels, f"{save}/in_labels.pth")
            torch.save(self.out_labels, f"{save}/out_labels.pth")

    @property
    def all_embeddings(self):
        return torch.cat([torch.cat(self.in_embeddings), torch.cat(self.out_embeddings)])


    def inner_product_attack(self, trials=1, subsample=None, normalize=True, use_pop_mean=False, whitening=False):
        task_inference_stat = []
        task_inference_label = []
        loop = tqdm(range(trials), desc="Running Inner Product Attack...", unit="trial", colour="red", position=0, leave=True)
        print("[Debug] Subsampling size: ", subsample)
        for trial in loop:
            for i in range(len(self.in_embeddings)):
                
                if subsample and subsample < len(self.in_embeddings[i]):
                    indices = random.sample(range(len(self.in_embeddings[i])), k=subsample)
                else:
                    indices = range(len(self.in_embeddings[i]))

                if use_pop_mean:
                    raise NotImplementedError("Set use_pop_mean = False")
                    # embs_to_atk = self.in_embeddings[i][indices] - self.population_means[self.in_labels[i][indices]]
                else:
                    embs_to_atk = self.in_embeddings[i][indices]
                
                if whitening: embs_to_atk = self.apply_whitening(embeddings=embs_to_atk, task_id=i, in_data=True)

                # Compute average inner product/cosine sim
                if normalize:
                    embs_to_atk = torch.nn.functional.normalize(embs_to_atk,)
                inners = (embs_to_atk@embs_to_atk.T).tril(diagonal=-1)
                mask = torch.ones_like(inners, dtype=bool).tril(diagonal=-1)

                stat = torch.abs(inners[mask]).mean()
                # stat = torch.pow(inners[mask], 2).mean()
                task_inference_stat.append(stat.item())
                task_inference_label.append(1)

            
            for i in range(len(self.out_embeddings)):
                
                if subsample and subsample < len(self.out_embeddings[i]):
                    indices = random.sample(range(len(self.out_embeddings[i])), k=subsample)
                else:
                    indices = range(len(self.out_embeddings[i]))

                if use_pop_mean:
                    raise NotImplementedError("Set use_pop_mean = False")
                    # embs_to_atk = self.out_embeddings[i][indices] - self.population_means[self.out_labels[i][indices]]
                else:
                    embs_to_atk = self.out_embeddings[i][indices]

                if whitening: embs_to_atk = embs_to_atk = self.apply_whitening(embeddings=embs_to_atk, task_id=i, in_data=False)

                # Compute average inner product/cosine sim
                if normalize:
                    embs_to_atk = torch.nn.functional.normalize(embs_to_atk,)
                # print(embs_to_atk.shape)
                inners = (embs_to_atk@embs_to_atk.T).tril(diagonal=-1)
                mask = torch.ones_like(inners, dtype=bool).tril(diagonal=-1)
                # stat = torch.pow(inners[mask], 2).mean()

                stat = torch.abs(inners[mask]).mean()
                task_inference_stat.append(stat.item())
                task_inference_label.append(0)

        return torch.tensor(task_inference_stat), torch.LongTensor(task_inference_label)

    def variance_attack(self, trials, subsample, use_pop_mean=False, whitening=False):
        
        task_inference_stat = []
        task_inference_label = []
        loop = tqdm(range(trials), desc="Running Coordinate-Wise Variance Attack...", unit="trial", colour="red")
        
        for trial in loop:
            for i in range(len(self.in_embeddings)):
                
                if subsample and subsample < len(self.in_embeddings[i]):
                    indices = random.sample(range(len(self.in_embeddings[i])), k=subsample)
                else:
                    indices = range(len(self.in_embeddings[i]))
    
                if use_pop_mean:
                    raise NotImplementedError("Set use_pop_mean = False")
                    # embs_to_atk = self.in_embeddings[i][indices] - self.population_means[self.in_labels[i][indices]]
                else:
                    embs_to_atk = self.in_embeddings[i][indices]

                if whitening: embs_to_atk = self.apply_whitening(embeddings=embs_to_atk, task_id=i, in_data=True)

                stat = embs_to_atk.T.cov().trace()

                task_inference_stat.append(stat.item())
                task_inference_label.append(1)

            
            for i in range(len(self.out_embeddings)):
                
                if subsample and subsample < len(self.out_embeddings[i]):
                    indices = random.sample(range(len(self.out_embeddings[i])), k=subsample)
                else:
                    indices = range(len(self.out_embeddings[i]))
                
                if use_pop_mean:
                    raise NotImplementedError("Set use_pop_mean = False")
                    # embs_to_atk = self.out_embeddings[i][indices] - self.population_means[self.out_labels[i][indices]]
                else:
                    embs_to_atk = self.out_embeddings[i][indices]

                if whitening: embs_to_atk = self.apply_whitening(embeddings=embs_to_atk, task_id=i, in_data=False)

                stat = embs_to_atk.T.cov().trace()


                task_inference_stat.append(stat.item())
                task_inference_label.append(0)
        return torch.tensor(task_inference_stat), torch.LongTensor(task_inference_label)
 
    def _compute_whitening_transform(self, task_id, in_data=True):
        if in_data:
            all_embeddings = torch.cat([
                torch.cat(self.in_embeddings[:task_id] + self.in_embeddings[task_id+1:]),
                torch.cat(self.out_embeddings)
            ])
        else:
            all_embeddings = torch.cat([
                torch.cat(self.in_embeddings),
                torch.cat(self.out_embeddings[:task_id] + self.out_embeddings[task_id+1:])
            ])

        # Compute mean for centering
        embedding_mean = torch.mean(all_embeddings, dim=0)
        
        # Compute covariance of already centered data
        ridge_lambda = 1e-8
        cov = all_embeddings.T.cov()

        to_reg = cov.clone()
        with torch.no_grad():
            while torch.linalg.cond(to_reg) > 500: 
                to_reg = (cov + (ridge_lambda * (cov.trace()/cov.shape[0]) * torch.eye(cov.shape[0]))).clone()
                ridge_lambda *= 10
                if ridge_lambda > 1: raise Exception("Cannot compute covariance")
            # print(f"[Debug] Exited at lambda={ridge_lambda} with condition number {torch.linalg.cond(to_reg):.2f}")

            regularized_cov = to_reg.clone()
            # Compute whitening transform
            eigenvalues, eigenvectors = torch.linalg.eigh(regularized_cov)
            D_sqrt_inv = torch.diag(1.0 / torch.sqrt(eigenvalues))
            whitening_transform = eigenvectors @ D_sqrt_inv @ eigenvectors.T

            return embedding_mean, whitening_transform
    
    def apply_whitening(self, embeddings, task_id, in_data=True):

        if in_data:
            mean, transform = self.whitening_transform_mean_in[task_id], self.whitening_transform_cov_in[task_id]
        else:
            mean, transform = self.whitening_transform_mean_out[task_id], self.whitening_transform_cov_out[task_id]
        
        return (embeddings - mean) @ transform