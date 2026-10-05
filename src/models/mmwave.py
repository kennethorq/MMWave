# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
MMWave - Multi-modal Recommender System
Reference:
    IEEE TKDE 2026: Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation

Reference Code:
    https://github.com/kennethorq/MMWave
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
import ptwt
import pywt

class MMWave(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMWave, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.audio_knn_k = config['audio_knn_k']

        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.visual_modal_weights = config['visual_modal_weights']
        self.text_modal_weights = config['text_modal_weights']
        self.audio_modal_weights = config['audio_modal_weights']
        self.fused_modal_weights = config['fused_modal_weights']

        self.wavelet_decom_level = config['wavelet_decom_level']
        self.wavelet = pywt.Wavelet('haar')
        self.padding_mode = 'zero'

        self.a_feat = getattr(self, 'a_feat', None)

        max_levels = []
        if self.v_feat is not None:
            image_feat_dim = self.v_feat.shape[1]
            max_image_level = pywt.dwt_max_level(image_feat_dim, self.wavelet.dec_len)
            max_levels.append(max_image_level)

        if self.t_feat is not None:
            text_feat_dim = self.t_feat.shape[1]
            max_text_level = pywt.dwt_max_level(text_feat_dim, self.wavelet.dec_len)
            max_levels.append(max_text_level)

        if self.a_feat is not None:
            audio_feat_dim = self.a_feat.shape[1]
            max_audio_level = pywt.dwt_max_level(audio_feat_dim, self.wavelet.dec_len)
            max_levels.append(max_audio_level)

        max_allowed_level = min(max_levels)

        if self.wavelet_decom_level > max_allowed_level:
            raise ValueError(
                f"wavelet_decom_level={self.wavelet_decom_level} is too large. "
                f"Maximum allowed level for the available modalities is {max_allowed_level}."
            )

        if self.v_feat is not None:
            self.image_approx_len = self._get_approx_len(self.v_feat.shape[1])
            self.image_complex_weight = nn.Parameter(
                torch.randn(1, self.image_approx_len, 2, dtype=torch.float32) * self.visual_modal_weights
            )

        if self.t_feat is not None:
            self.text_approx_len = self._get_approx_len(self.t_feat.shape[1])
            self.text_complex_weight = nn.Parameter(
                torch.randn(1, self.text_approx_len, 2, dtype=torch.float32) * self.text_modal_weights
            )

        if self.a_feat is not None:
            self.audio_approx_len = self._get_approx_len(self.a_feat.shape[1])
            self.audio_complex_weight = nn.Parameter(
                torch.randn(1, self.audio_approx_len, 2, dtype=torch.float32) * self.audio_modal_weights
            )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))
        audio_adj_file = os.path.join(dataset_path, 'audio_adj_{}_{}.pt'.format(self.audio_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False) 
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)                            
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()                                             

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)              
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)                                
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')   
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        if self.a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)                   
            if os.path.exists(audio_adj_file):
                audio_adj = torch.load(audio_adj_file)                      
            else:
                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj = build_knn_normalized_graph(audio_adj, topk=self.audio_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(audio_adj, audio_adj_file)
            self.audio_original_adj = audio_adj.cuda() 

        self.fusion_adj = self.eigen_decomposition_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        if self.a_feat is not None:
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common_mod = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        num_modalities = sum(feat is not None for feat in [self.v_feat, self.t_feat, self.a_feat])
        self.query_common_fused = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, num_modalities, bias=False)
        )

        if self.v_feat is not None:
            self.gate_v_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))
            self.gate_image_prefer_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))

        if self.t_feat is not None:
            self.gate_t_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))
            self.gate_text_prefer_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))

        if self.a_feat is not None:
            self.gate_a_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))
            self.gate_audio_prefer_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))

        self.gate_f_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))
        self.gate_fusion_prefer_weight = nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim))

        if self.v_feat is not None: nn.init.xavier_uniform_(self.gate_v_weight)
        if self.t_feat is not None: nn.init.xavier_uniform_(self.gate_t_weight)
        nn.init.xavier_uniform_(self.gate_f_weight)
        if self.v_feat is not None: nn.init.xavier_uniform_(self.gate_image_prefer_weight)
        if self.t_feat is not None: nn.init.xavier_uniform_(self.gate_text_prefer_weight)
        nn.init.xavier_uniform_(self.gate_fusion_prefer_weight)

        if self.a_feat is not None:
            nn.init.xavier_uniform_(self.gate_a_weight)
            nn.init.xavier_uniform_(self.gate_audio_prefer_weight)

        self.gate_v = nn.Sequential(
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Sigmoid()
        )
        self.gate_f = nn.Sequential(
            nn.Sigmoid()
        )
        self.gate_image_prefer = nn.Sequential(
            nn.Sigmoid()
        )
        self.gate_text_prefer = nn.Sequential(
            nn.Sigmoid()
        )        
        self.gate_audio_prefer = nn.Sequential(
            nn.Sigmoid()
        )        
        self.gate_fusion_prefer = nn.Sequential(
            nn.Sigmoid()
        )        

        num_modalities = sum(feat is not None for feat in [self.v_feat, self.t_feat, self.a_feat])

        if num_modalities == 2:
            self.alpha = nn.Parameter(torch.full((self.n_items, self.embedding_dim), 0.5))

        elif num_modalities == 3:
            self.modal_logits = nn.Parameter(torch.zeros(self.n_items, self.embedding_dim, 3))

    def pre_epoch_processing(self):
        pass

    def _get_approx_len(self, feat_dim):
        """
        Compute the length of approximation coefficients produced by ptwt.wavedec
        for the current wavelet/padding/level setting.
        """
        dummy = torch.zeros(1, feat_dim, dtype=torch.float32)
        coeffs = ptwt.wavedec(
            dummy,
            self.wavelet,
            mode=self.padding_mode,
            level=self.wavelet_decom_level
        )
        return coeffs[0].shape[1]

    def eigen_decomposition_fusion(self):
        with torch.no_grad():
            adjs = []

            if self.v_feat is not None:
                adjs.append(self.image_original_adj.cpu().to_dense())

            if self.t_feat is not None:
                adjs.append(self.text_original_adj.cpu().to_dense())

            if self.a_feat is not None:
                adjs.append(self.audio_original_adj.cpu().to_dense())

            eigvals_norm = []
            for adj in adjs:
                eigvals = torch.linalg.eigvalsh(adj)
                eigvals_norm.append(eigvals / eigvals.max())

            weighted_adjs = []
            for i, adj in enumerate(adjs):
                other_eigvals = sum(eigvals_norm[j] for j in range(len(adjs)) if j != i) / (len(adjs) - 1)
                weighted_adjs.append(adj * other_eigvals.unsqueeze(0))

            fused_adj = sum(weighted_adjs) / len(weighted_adjs)
            fused_adj = torch.clamp(fused_adj, min=0)

            fused_adj_sparse = fused_adj.to_sparse().coalesce()

            del adjs, eigvals_norm, weighted_adjs, fused_adj

        return fused_adj_sparse.cuda()


    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def wavelet_extraction(self, image_embeds, text_embeds, audio_embeds):
        wavelet = pywt.Wavelet('haar')
        padding_mode = 'zero'

        image_conv, text_conv, audio_conv = None, None, None

        if image_embeds is not None:
            image_wavelet = ptwt.wavedec(image_embeds, wavelet, mode=padding_mode, level=self.wavelet_decom_level)
            image_approx = image_wavelet[0]
            image_conv_wavelet = torch.mul(image_approx, torch.view_as_complex(self.image_complex_weight))
            image_real = image_conv_wavelet.real
            zero_coeffs_image = [torch.zeros_like(image_wavelet[i]).to(self.device) for i in range(1, len(image_wavelet))]
            image_conv = ptwt.waverec([image_real] + zero_coeffs_image, wavelet)

        if text_embeds is not None:
            text_wavelet = ptwt.wavedec(text_embeds, wavelet, mode=padding_mode, level=self.wavelet_decom_level)
            text_approx = text_wavelet[0]
            text_conv_wavelet = torch.mul(text_approx, torch.view_as_complex(self.text_complex_weight))
            text_real = text_conv_wavelet.real
            zero_coeffs_text = [torch.zeros_like(text_wavelet[i]).to(self.device) for i in range(1, len(text_wavelet))]
            text_conv = ptwt.waverec([text_real] + zero_coeffs_text, wavelet)

        if audio_embeds is not None:
            audio_wavelet = ptwt.wavedec(audio_embeds, wavelet, mode=padding_mode, level=self.wavelet_decom_level)
            audio_approx = audio_wavelet[0]
            audio_conv_wavelet = torch.mul(audio_approx, torch.view_as_complex(self.audio_complex_weight))
            audio_real = audio_conv_wavelet.real
            zero_coeffs_audio = [torch.zeros_like(audio_wavelet[i]).to(self.device) for i in range(1, len(audio_wavelet))]
            audio_conv = ptwt.waverec([audio_real] + zero_coeffs_audio, wavelet)

        return image_conv, text_conv, audio_conv
    
    def forward(self, adj, train=False):

        image_feats = self.image_embedding.weight if self.v_feat is not None else None
        text_feats = self.text_embedding.weight if self.t_feat is not None else None
        audio_feats = self.audio_embedding.weight if self.a_feat is not None else None

        #   Wavelet Modality Mixer
        denoised_image, denoised_text, denoised_audio = self.wavelet_extraction(image_feats, text_feats, audio_feats)

        if self.v_feat is not None:
            denoised_image = self.image_trs(denoised_image)
        if self.t_feat is not None:
            denoised_text = self.text_trs(denoised_text)
        if self.a_feat is not None:
            denoised_audio = self.audio_trs(denoised_audio)

        #   Attentive fusion
        if self.v_feat is not None and self.t_feat is not None and self.a_feat is None:
            alpha_weight = torch.sigmoid(self.alpha)
            fusion_conv = alpha_weight * denoised_image + (1 - alpha_weight) * denoised_text

        elif self.v_feat is not None and self.a_feat is not None and self.t_feat is None:
            alpha_weight = torch.sigmoid(self.alpha)
            fusion_conv = alpha_weight * denoised_image + (1 - alpha_weight) * denoised_audio

        elif self.t_feat is not None and self.a_feat is not None and self.v_feat is None:
            alpha_weight = torch.sigmoid(self.alpha)
            fusion_conv = alpha_weight * denoised_text + (1 - alpha_weight) * denoised_audio

        else:
            modal_weights = torch.softmax(self.modal_logits, dim=-1)
            fusion_conv = modal_weights[:, :, 0] * denoised_image + \
                        modal_weights[:, :, 1] * denoised_text + \
                        modal_weights[:, :, 2] * denoised_audio
    
        image_conv = denoised_image
        text_conv = denoised_text
        audio_conv = denoised_audio

        if self.v_feat is not None:
            image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(torch.matmul(image_conv, self.gate_v_weight)))
        if self.t_feat is not None:
            text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(torch.matmul(text_conv, self.gate_t_weight)))
        if self.a_feat is not None:
            audio_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_a(torch.matmul(audio_conv, self.gate_a_weight)))

        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(torch.matmul(fusion_conv, self.gate_f_weight)))

        #   Structure-aware Multi-modal Graph Learning
        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Multi-modal Views
        #   Image-view
        if self.v_feat is not None:
            if self.sparse:
                for i in range(self.n_layers):
                    image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
            else:
                for i in range(self.n_layers):
                    image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
            image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
            image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.t_feat is not None:
            if self.sparse:
                for i in range(self.n_layers):
                    text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
            else:
                for i in range(self.n_layers):
                    text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
            text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
            text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Audio-view
        if self.a_feat is not None:
            if self.sparse:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.sparse.mm(self.audio_original_adj, audio_item_embeds)
            else:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.mm(self.audio_original_adj, audio_item_embeds)
            audio_user_embeds = torch.sparse.mm(self.R, audio_item_embeds)
            audio_embeds = torch.cat([audio_user_embeds, audio_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        # Adaptive-Modality Regulator
        att_common_list, modality_embeds = [], []

        if self.v_feat is not None:
            att_common_list.append(self.query_common_mod(image_embeds))
            modality_embeds.append(image_embeds)

        if self.t_feat is not None:
            att_common_list.append(self.query_common_mod(text_embeds))
            modality_embeds.append(text_embeds)

        if self.a_feat is not None:
            att_common_list.append(self.query_common_mod(audio_embeds))
            modality_embeds.append(audio_embeds)

        att_common = self.softmax(torch.cat(att_common_list, dim=-1))

        att_f = self.query_common_fused(fusion_embeds)
        att_fused = self.softmax(att_f)

        common_fused_embeds = sum(
            att_common[:, i].unsqueeze(dim=1) * fusion_embeds
            for i in range(len(modality_embeds))
        ) / len(modality_embeds)

        uni_embeds = [
            att_fused[:, i].unsqueeze(dim=1) * modality_embeds[i]
            for i in range(len(modality_embeds))
        ]

        sep_embeds = []
        idx = 0

        if self.v_feat is not None:
            image_prefer = self.gate_image_prefer(torch.matmul(content_embeds, self.gate_image_prefer_weight))
            image_prefer = self.dropout(image_prefer)
            sep_image_embeds = torch.multiply(image_prefer, uni_embeds[idx])
            sep_embeds.append(sep_image_embeds)
            idx += 1

        if self.t_feat is not None:
            text_prefer = self.gate_text_prefer(torch.matmul(content_embeds, self.gate_text_prefer_weight))
            text_prefer = self.dropout(text_prefer)
            sep_text_embeds = torch.multiply(text_prefer, uni_embeds[idx])
            sep_embeds.append(sep_text_embeds)
            idx += 1

        if self.a_feat is not None:
            audio_prefer = self.gate_audio_prefer(torch.matmul(content_embeds, self.gate_audio_prefer_weight))
            audio_prefer = self.dropout(audio_prefer)
            sep_audio_embeds = torch.multiply(audio_prefer, uni_embeds[idx])
            sep_embeds.append(sep_audio_embeds)

        fusion_prefer = self.gate_fusion_prefer(torch.matmul(content_embeds, self.gate_fusion_prefer_weight))
        fusion_prefer = self.dropout(fusion_prefer)
        fusion_embeds = torch.multiply(fusion_prefer, common_fused_embeds)

        sep_embeds.append(fusion_embeds)
        side_embeds = torch.mean(torch.stack(sep_embeds), dim=0)

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores