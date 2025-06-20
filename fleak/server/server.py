import copy
import numpy as np
import torch
from collections import OrderedDict

from ..utils.lowrank_decomposition import reconstruct_from_low_rank,reconstruct_from_low_rank_nmf,reconstruct_from_QR,check_validity,reconstruct_from_lr_cnn
from ..utils.getsize import get_ordereddict_size, format_size

from ..attack import dlg, idlg
from ..attack import ig_single, ig_multi
from ..attack import invert_linear_layer
from ..attack import grnn
from ..attack import ggl
from ..attack import cpa
from ..attack import dlf
from ..attack import ggi
from ..attack.label import label_count_restoration
from ..attack.label import label_count_to_label


class Server:

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        test_loader=None,
        device=None
    ):
        # server info
        self.server_id = server_id
        self.server_group = server_group
        self.device = device
        self.selected_clients = None
        self.updates = []
        self.cur_round = 0
        # model & data
        self.global_model = global_model.to(self.device)
        self.test_loader = test_loader

    @property
    def model_size(self):
        return sum([p.numel() * p.element_size() for p in self.global_model.state_dict().values()])

    def select_clients(self, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def train_eval(self, clients=None, set_to_use='test'):
        if clients is None:
            clients = self.selected_clients
        eval_correct = 0
        eval_total = 0
        for c in clients:
            c.synchronize(self.cur_round, self.global_model.state_dict())
            eval_cor, eval_tot = c.evaluate(set_to_use=set_to_use)
            eval_correct += eval_cor
            eval_total += eval_tot
            # train and update client model
            c_id, num_samples, update_raw, update = c.train()
            # update client round
            self.updates.append((c_id, num_samples, update_raw, update))
            # # train and update client model
            # c_id, num_samples, update = c.train()
            # # update client round
            # self.updates.append((c_id, num_samples, update))
        eval_accuracy = eval_correct / eval_total
        print('Round %d: ' % self.cur_round + set_to_use + ' accuracy %.4f' % eval_accuracy)
        # update communication round
        self.cur_round += 1
        return eval_accuracy

    def train(self, clients=None):
        # just for training
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.synchronize(self.cur_round, self.global_model.state_dict())
            # train and update client model
            c_id, num_samples, update_raw, update = c.train()
            # gather client uploads into a buffer
            self.updates.append((c_id, num_samples, update_raw, update))
        # update communication round
        self.cur_round += 1

    def evaluate(self, clients=None, set_to_use='test'):
        if clients is None:
            clients = self.selected_clients
        correct = 0
        total = 0
        for i, c in enumerate(clients):
            c.synchronize(self.cur_round, self.global_model.state_dict())
            corr, tot = c.evaluate(set_to_use=set_to_use)
            correct += corr
            total += tot
        accuracy = correct / total
        print('Round %d: ' % self.cur_round + set_to_use + ' accuracy %.4f' % accuracy)
        return accuracy

    def federated_averaging(self):
        total_samples = np.sum([update[1] for update in self.updates])
        averaged_soln = copy.deepcopy(self.updates[0][2])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    if i == 0:
                        total_bytes2 = get_ordereddict_size(self.updates[i][-2])
                        print(f" {format_size(total_bytes2)}")
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += self.updates[i][2][key] * self.updates[i][1] / total_samples
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []

    def federated_averaging_1(self):
        total_samples = np.sum([update[1] for update in self.updates])
        averaged_soln = copy.deepcopy(self.updates[0][-1])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += self.updates[i][2][key] * self.updates[i][1] / total_samples
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []

    def federated_averaging_low_rank_a(self):
        total_samples = np.sum([update[1] for update in self.updates])
        low_rank_param_list=[]
        low_rank_dict = OrderedDict()
        averaged_soln = copy.deepcopy(self.global_model.state_dict())
        for i in range(len(self.updates)):
            ## 加权
            for key, value in self.updates[i][-1].items():
                if "weight" in key:
                    low_rank_params = tuple(value[j] * self.updates[i][1] / total_samples for j in range(len(value)))
                    low_rank_dict.update({key:low_rank_params})
                elif "bias" in key:
                    low_rank_params = torch.from_numpy(value * self.updates[i][1] / total_samples)
                    low_rank_dict.update({key: low_rank_params})
            low_rank_param_list.append(low_rank_dict)
        aggregated_low_rank_params = {}
        for i in range(len(self.updates)):
            aggregated_low_rank_params[key] = [
                sum(params[key][i] for params in low_rank_param_list)
                for i in range(3)
            ]
        # for key in low_rank_param_list[0].keys():
        #     aggregated_low_rank_params[key] = [
        #         sum(params[key][i] for params in low_rank_param_list)
        #         for i in range(3)
        #     ]
        for key in aggregated_low_rank_params.keys():
            if "weight" in key:
                U, S, Vt = aggregated_low_rank_params[key]
                reconstructed_weight = reconstruct_from_low_rank(U, S, Vt)
                averaged_soln[key] = reconstructed_weight
            else:
                averaged_soln[key] = aggregated_low_rank_params[key]
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []


    def federated_averaging_low_rank_w(self):
        total_samples = np.sum([update[1] for update in self.updates])
        low_rank_param_list=[]
        low_rank_dict = OrderedDict()
        recover_matrix = OrderedDict()
        recover_matrix_full_list = []
        ## recover
        for i in range(len(self.updates)):
            # if i == 0:
            #     total_bytes1 = get_ordereddict_size(self.updates[i][-1])
            #     total_bytes2 = get_ordereddict_size(self.updates[i][-2])
            #     print(total_bytes1)
            #     print(total_bytes2)
            #     print(f" {format_size(total_bytes1)}")
            #     print(f" {format_size(total_bytes2)}")
            first_weight_processed = False
            for key in self.updates[i][-1].keys():
                if "weight" in key and not first_weight_processed:
                    (U, s_u), S, (Vt, s_v), U_residual, V_residual = self.updates[i][-1][key]
                    reconstructed_weight = reconstruct_from_low_rank(U, S, Vt, s_u, s_v, U_residual, V_residual)
                    # U, S, Vt = self.updates[i][-1][key]
                    # reconstructed_weight = reconstruct_from_low_rank(U, S, Vt)
                    if "conv" in key:
                        first_layer = self.global_model.conv1
                        reconstructed_weight = reconstructed_weight.reshape(first_layer.weight.size())
                    dict_new = {key : reconstructed_weight}
                    recover_matrix.update(dict_new)
                    first_weight_processed = True
                else:
                    dict_new = {key: self.updates[i][-1][key]}
                    recover_matrix.update(dict_new)

            recover_matrix_full_list.append(copy.deepcopy(recover_matrix))
        ## aggregation
        averaged_soln = copy.deepcopy(recover_matrix_full_list[0])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += recover_matrix_full_list[i][key] * self.updates[i][1] / total_samples

        # update global model
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []



    def federated_averaging_low_rank_NMF(self):
        total_samples = np.sum([update[1] for update in self.updates])
        recover_matrix = OrderedDict()
        recover_matrix_full_list = []
        ## recover
        for i in range(len(self.updates)):
            for key in self.updates[i][-1].keys():
                if "weight" in key:
                    w, h = self.updates[i][-1][key]
                    reconstructed_weight = reconstruct_from_low_rank_nmf(w,h)
                    dict_new = {key : reconstructed_weight}
                    recover_matrix.update(dict_new)
                elif "bias" in key:
                    dict_new = {key : torch.from_numpy(self.updates[i][-1][key])}
                    recover_matrix.update(dict_new)
            recover_matrix_full_list.append(recover_matrix)
        ## aggregation
        averaged_soln = copy.deepcopy(recover_matrix_full_list[0])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += recover_matrix_full_list[i][key] * self.updates[i][1] / total_samples
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []


    def federated_averaging_QR_mask(self):
        total_samples = np.sum([update[1] for update in self.updates])
        recover_matrix = OrderedDict()
        recover_matrix_full_list = []
        ## recover
        for i in range(len(self.updates)):
            for key in self.updates[i][-1].keys():
                if "fc2.weight" in key:
                # if "fc2.weight" in key:
                    Q,R = self.updates[i][-1][key]
                    # check_validity(Q,R)
                    reconstructed_weight = reconstruct_from_QR(Q, R)
                    # reconstructed_weight += 1e-6 * np.eye(*reconstructed_weight.shape[:1], M=reconstructed_weight.shape[1])
                    dict_new = {key : reconstructed_weight}
                    recover_matrix.update(dict_new)
                else:
                    dict_new = {key:self.updates[i][-1][key]}
                    recover_matrix.update(dict_new)

            recover_matrix_full_list.append(recover_matrix)
        ## aggregation
        averaged_soln = copy.deepcopy(recover_matrix_full_list[0])
        for key in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            for i in range(len(self.updates)):
                if i == 0:
                    averaged_soln[key] = averaged_soln[key] * self.updates[i][1] / total_samples
                else:
                    averaged_soln[key] += recover_matrix_full_list[i][key] * self.updates[i][1] / total_samples
        # update global model
        self.global_model.load_state_dict(averaged_soln)
        # clear uploads buffer
        self.updates = []

    def save_model(self, path):
        # Save server model
        torch.save(self.global_model.state_dict(), path)


class ServerAttacker(Server):

    def __init__(
        self,
        server_id=None,
        server_group=None,
        global_model=None,
        generator=None,
        test_loader=None,
        dummy=None,
        device=None
    ):
        super().__init__(
            server_id=server_id,
            server_group=server_group,
            global_model=global_model,
            test_loader=test_loader,
            device=device
        )
        if generator is not None:
            self.generator = generator.to(self.device)
        self.dummy = dummy

    def extract_gradients(self, local_params, iterations=1, lr=1):
        """ Extract the gradients of any client model

        The ground-truth gradients are approximated by (W0 - WT) / (T * lr)
        This is only valid for SGD without momentum

        Using named_parameters to avoid incorrect computation with running statistics
        Caution: .detach() is adopted here to cut off the grad_fn

        :param local_params: client model parameters
        :param iterations: number of update iterations
        :param lr: local learning rate
        :return:
        """
        diffs = [((v - local_params[k]) / (iterations * lr)).detach() for k, v in self.global_model.named_parameters()]
        return diffs

    def attack(self, args, clients=None):
        """
        Randomly select a client to infer its private data

        :param args: attack arguments
        :param clients: possible clients
        :return: reconstructed data and labels
        """
        if clients is None:
            clients = self.selected_clients
        tot_clients = len(clients)
        # randomly select one client to attack
        attack_cid = np.random.randint(0, tot_clients)

        if args.attack == "gdl":
            # Towards General Deep Leakage in Federated Learning https://arxiv.org/pdf/2110.09074
            # reconstruction from weights locally updated multiple steps
            iterations, lr = (self.updates[attack_cid][1] / args.batch_size) * args.local_epochs, args.lr
        elif args.attack == "dlf":
            # Data Leakage in Federated Averaging https://openreview.net/pdf?id=e7A0B99zJf
            iterations, lr = 1, args.lr
        else:
            iterations, lr = 1, 1

        local_grads = self.extract_gradients(self.updates[attack_cid][-2], iterations=iterations, lr=lr)


        # local_parameters_matrix = OrderedDict()
        # first_weight_processed = False
        # for key in self.updates[attack_cid][-1].keys():
        #     if "weight" in key and not first_weight_processed:
        #         (U, s_u), S, (Vt, s_v), U_residual, V_residual = self.updates[attack_cid][-1][key]
        #         reconstructed_weight = reconstruct_from_low_rank(U, S, Vt, s_u, s_v, U_residual, V_residual)
        #         # U, S, Vt = self.updates[attack_cid][-1][key]
        #         # reconstructed_weight = reconstruct_from_low_rank(U, S, Vt)
        #         if "conv" in key:
        #             first_layer = self.global_model.conv1
        #             reconstructed_weight = reconstructed_weight.reshape(first_layer.weight.size())
        #         # Q,R = self.updates[attack_cid][-1][key]
        #         # reconstructed_weight = reconstruct_from_QR(Q,R)
        #         dict_new = {key: reconstructed_weight}
        #         local_parameters_matrix.update(dict_new)
        #         first_weight_processed = True
        #     else:
        #         dict_new = {key: self.updates[attack_cid][-1][key]}
        #         local_parameters_matrix.update(dict_new)
        # local_parameters_matrix = {key: local_parameters_matrix[key].to(self.device) for key in local_parameters_matrix}
        #
        # local_grads = self.extract_gradients(local_parameters_matrix, iterations=iterations, lr=lr)



        if args.attack == "dlg":
            dlg(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                device=self.device
            )
        elif args.attack == "idlg":
            idlg(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                device=self.device
            )
        elif args.attack == "ig_single":
            ig_single(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                tv=args.tv,
                device=self.device
            )
        elif args.attack == "ggi":
            ggi(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                tv=args.tv,
                device=self.device
            )
        elif args.attack == "ig_multi":
            ig_multi(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                local_epochs=args.local_epochs,
                local_lr=args.lr,
                tv=args.tv,
                device=self.device
            )
        elif args.attack == "rtf":
            invert_linear_layer(gt_grads=local_grads, dummy=self.dummy)
        elif args.attack == "ggl":
            ggl(
                model=self.global_model,
                generator=self.generator,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                device=self.device
            )
        elif args.attack == "grnn":
            grnn(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                tv=args.tv,
                device=self.device
            )
        elif args.attack == "cpa":
            cpa(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                fi_lr=args.fi_lr,
                decor=args.decor,
                T=args.T,
                tv=args.tv,
                nv=args.nv,
                l1=args.l1,
                fi=args.fi,
                device=self.device
            )
        elif args.attack == "dlf":
            # restore batch labels
            label_counts = label_count_restoration(
                model=self.global_model,
                o_state=self.global_model.state_dict(),
                n_state=self.updates[attack_cid][-2],
                # n_state=local_parameters_matrix,
                deltaW=local_grads,
                dummy=self.dummy,
                local_data_size=self.updates[attack_cid][1],
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                device=self.device
            )
            dummy_labels = label_count_to_label(label_counts, self.device)
            dlf(
                model=self.global_model,
                gt_grads=local_grads,
                dummy=self.dummy,
                labels=dummy_labels,
                rec_epochs=args.rec_epochs,
                rec_lr=args.rec_lr,
                epochs=args.local_epochs,
                lr=args.lr,
                data_size=self.updates[attack_cid][1],
                batch_size=args.batch_size,
                tv=args.tv,
                reg_clip=args.reg_clip,
                reg_reorder=args.reg_reorder,
                device=args.device
            )
        else:
            raise ValueError("Unexpected {} Attack Type.".format(args.attack))