import collections

import numpy as np
import torch
import torch.nn.functional as F


def where(cond, x_1, x_2):
    """
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/9
    :param cond:
    :param x_1:
    :param x_2:
    :return:
    """
    cond = cond.type_as(x_1)
    return (cond * x_1) + ((1 - cond) * x_2)


class PolynomialNormalization(object):
    """Dividing by the length (raised to some power (default 0.6))"""

    def __init__(self, alpha=0.6, apply_during_search=True):
        self.alpha = alpha
        self.apply_during_search = apply_during_search

    def lp(self, len):
        return pow(5 + len, self.alpha) / pow(5 + 1, self.alpha)

    def normalize_completed(self, completed_hyps, src_length=None):
        if not self.apply_during_search:
            for hyp in completed_hyps:
                hyp.score /= pow(len(hyp.id_list), self.m)

    def normalize_partial(self, score_so_far, score_to_add, new_len):
        if self.apply_during_search:
            return (score_so_far * self.lp(new_len - 1) + score_to_add) / self.lp(new_len)
        else:
            return score_so_far + score_to_add


def update_beam_state(outs, total_score, topk, topk_score,
                      eos_id, alpha, tokens_enc, z_block, types):
    full = outs.size()[0]
    prev_full, k = topk.size()
    batch = full // k
    prev_k = prev_full // batch
    assert (prev_k in [1, k])

    if total_score is None:
        total_score = topk_score
    else:
        is_end = torch.max(outs == eos_id, dim=1)[0]
        is_end = is_end.view(-1, 1).expand_as(topk_score)
        bias = torch.zeros_like(topk_score).type(z_block.type())
        bias[:, 1:] = -10000.  # remove ended cands except for a consequence

        obj = PolynomialNormalization(alpha=alpha)
        normalized_total_score = obj.normalize_partial(total_score[:, None],
                                                       topk_score,
                                                       outs.size()[1])
        # Use torch.where in v0.4
        total_score = where(is_end,
                            total_score[:, None] + bias,
                            normalized_total_score)

        # total_score = total_score.data
        assert (torch.max(total_score) < 0.)

        # Use torch.where in v0.4
        topk = where(is_end,
                     torch.LongTensor([eos_id]).cuda(),
                     topk)  # this is not required

    total_score = total_score.view((prev_full // prev_k, prev_k * k))
    total_topk_score, argtopk = torch.topk(total_score, k)

    assert (argtopk.size() == (prev_full // prev_k, k))
    assert (total_topk_score.size() == (prev_full // prev_k, k))

    total_topk = topk.take(argtopk + (torch.arange(prev_full // prev_k)[:, None] * prev_k * k).cuda())
    total_topk = total_topk.view((full,))
    total_topk_score = total_topk_score.view((full,))
    argtopk = argtopk // k + (torch.arange(prev_full // prev_k)[:, None] * prev_k).cuda()
    argtopk = argtopk.view((full,))

    xss = torch.split(tokens_enc, 1)
    tokens_enc = torch.cat([xss[i] for i in argtopk])

    zss = torch.split(z_block, 1)
    z_block = torch.cat([zss[i] for i in argtopk])

    yss = torch.split(types, 1)
    types = torch.cat([yss[i] for i in argtopk])

    outs = torch.split(outs, 1)
    outs = torch.cat([outs[i] for i in argtopk])
    outs = torch.cat([outs, total_topk[:, None]], dim=1).cuda()

    return outs, total_topk_score, tokens_enc, z_block, types


def finish_beam(outs, total_score, batchsize, eos_id):
    k = outs.shape[0] // batchsize
    result_batch = collections.defaultdict(lambda: {'outs': [], 'score': -np.inf})
    for i in range(batchsize):
        for j in range(k):
            score = total_score[i * k + j]
            if result_batch[i]['score'] < score:
                out = outs[i * k + j].tolist()
                if eos_id in out:
                    out = out[:out.index(eos_id)]
                result_batch[i] = {'outs': out, 'score': score}

    result_batch = [result for i, result in sorted(result_batch.items(), key=lambda x: x[0])]

    id_list, score_list = [], []
    for item in result_batch:
        id_list.append(item['outs'])
        score_list.append(item['score'])
    return id_list, score_list


class BeamSearch(object):
    def __init__(self, max_decode_len, bos_id, eos_id, beam_size=5, alpha=0.6, topk_evidence=-1):
        self.max_decode_length = max_decode_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.k = beam_size
        self.alpha = alpha
        assert topk_evidence >= 1, "this code is customized for retrieval tasks"

    def generate_output(self, model, query_uid, query_ids_bert, query_types, query_mask_bert, query_ids_t5, query_ids_t5_len):
        batch, x_length = query_ids_bert.shape
        bos_array = np.array([[self.bos_id]] * batch, dtype=np.int64)
        y_block = torch.LongTensor(bos_array).cuda()

        outs = torch.LongTensor([[self.bos_id]] * batch * self.k).cuda()
        total_score = None
        all_query_context_hidden_states = None
        all_query_context_ids_unflat = None
        topk_log_probs = None

        for i in range(self.max_decode_length):
            logits, topk_log_probs, all_query_context_hidden_states, all_query_context_ids_unflat = \
                model(query_uid,
                      query_ids_bert,
                      query_types,
                      query_mask_bert,
                      query_ids_t5,
                      query_ids_t5_len,
                      y_block,
                      all_query_context_hidden_states=all_query_context_hidden_states,
                      all_query_context_ids_unflat=all_query_context_ids_unflat,
                      topk_log_probs=topk_log_probs)

            topk_score, topk = torch.topk(F.log_softmax(logits[:, -1, :], dim=1),
                                          self.k)
            assert (torch.max(topk_score) <= 0.)

            outs, total_score, all_query_context_ids_unflat, all_query_context_hidden_states, topk_log_probs = \
                update_beam_state(outs,
                                  total_score,
                                  topk,
                                  topk_score,
                                  self.eos_id,
                                  self.alpha,
                                  all_query_context_ids_unflat,
                                  all_query_context_hidden_states,
                                  topk_log_probs)
            assert (torch.max(total_score < 0.)), i
            y_block = outs

            if torch.max(outs == self.eos_id, 1)[0].sum() == outs.shape[0]:
                break  # all cands meet eos, end the loop

        id_list, score_list = finish_beam(outs[:, 1:], total_score, batch, self.eos_id)
        return id_list


class SampleOrGreedySearch(object):
    def __init__(self, max_decode_len, bos_id, eos_id, sample=False, topk_evidence=-1):
        self.max_decode_length = max_decode_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.sample = sample
        assert topk_evidence >= 1, "this code is customized for retrieval tasks"

    def generate_output(self, model, query_uid, query_ids_bert, query_types, query_mask_bert, query_ids_t5, query_ids_t5_len):
        batch, x_length = query_ids_bert.shape
        y_block = np.full((batch, 1), self.bos_id, dtype=np.int64)
        eos_flags = np.zeros((batch,), dtype=np.int32)

        y_block = torch.LongTensor(y_block).cuda()
        result = []
        all_query_context_hidden_states = None
        all_query_context_ids_unflat = None
        topk_log_probs = None

        for i in range(self.max_decode_length):
            logits, topk_log_probs, all_query_context_hidden_states, all_query_context_ids_unflat = \
                model(query_uid,
                      query_ids_bert,
                      query_types,
                      query_mask_bert,
                      query_ids_t5,
                      query_ids_t5_len,
                      y_block,
                      all_query_context_hidden_states=all_query_context_hidden_states,
                      all_query_context_ids_unflat=all_query_context_ids_unflat,
                      topk_log_probs=topk_log_probs)

            if self.sample:
                ys = torch.multinomial(F.softmax(logits[:, -1, :],
                                                 dim=1),
                                       num_samples=1).squeeze()
                if ys.numel() == 1:
                    ys = ys.unsqueeze(-1)
            else:
                log_score, ys = torch.max(F.log_softmax(logits[:, -1, :], dim=1), dim=1)

            y_block = torch.cat([y_block.detach(), ys[:, None]], dim=1)
            ys = ys.data.cpu().numpy()
            result.append(ys)
            eos_flags += (ys == self.eos_id)
            if np.all(eos_flags):
                break

        result = np.stack(result).T
        # Remove EOS tags
        outs = []
        for j, y in enumerate(result):
            inds = np.argwhere(y == self.eos_id)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y.tolist())

        return outs
