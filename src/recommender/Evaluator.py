import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time
import datetime
import heapq
import math

_feed_dict_test = None
_feed_dict_validation = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data, val=False):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    feed_dicts_test = pool.map(_evaluate_input_list_test, range(_dataset.num_users))
    pool.close()
    pool.join()

    if val:
        pool = Pool(cpu_count() - 1)
        feed_dicts_validation = pool.map(_evaluate_input_list_validation, range(_dataset.num_users))
        pool.close()
        pool.join()
        return feed_dicts_validation, feed_dicts_test

    return feed_dicts_test


def _evaluate_input_list_test(user):
    test_items = _dataset.test_list[user]

    if len(test_items) > 0:
        item_input = set(range(_dataset.num_items)) - set(_dataset.training_list[user])

        for test_item in test_items:
            if test_item in item_input:
                item_input.remove(test_item)

        item_input = list(item_input)

        for test_item in test_items:
            item_input.append(test_item)

        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    else:
        print('User {} has no test list!'.format(user))
        return 0, 0


def _evaluate_input_list_validation(user):
    validation_items = _dataset.validation_list[user]

    if len(validation_items) > 0:
        item_input = set(range(_dataset.num_items)) - set(_dataset.training_list[user])

        for validation_item in validation_items:
            if validation_item in item_input:
                item_input.remove(validation_item)

        item_input = list(item_input)

        for validation_item in validation_items:
            item_input.append(validation_item)

        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input
    else:
        print('User {} has no test list!'.format(user))
        return 0, 0


def _eval_by_user(user, curr_pred, val=False):
    # get predictions of data in test set
    if val:
        user_input, item_input = _feed_dict_validation[user]
    else:
        user_input, item_input = _feed_dict_test[user]
    if type(user_input) != np.ndarray:
        return ()

    # AREA UNDER CURVE (AUC)
    predictions = curr_pred[list(item_input.reshape(-1))]
    neg_predict, pos_predict = predictions[:-len(_dataset.test_list[user] if not val else _dataset.validation_list[user])], \
                               predictions[-len(_dataset.test_list[user] if not val else _dataset.validation_list[user]):]

    position = 0
    for t in range(len(_dataset.test_list[user] if not val else _dataset.validation_list[user])):
        position += (neg_predict >= pos_predict[t]).sum()

    auc = 1 - (position / (len(neg_predict) * len(pos_predict)))
    # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    # HIT RATIO (HR)
    item_score = {}
    for i in list(item_input.reshape(-1)):
        item_score[i] = curr_pred[i]

    k_max_item_score = heapq.nlargest(_K, item_score, key=item_score.get)

    r = []
    for i in k_max_item_score:
        if i in item_input[-len(_dataset.test_list[user] if not val else _dataset.validation_list[user]):]:
            r.append(1)
        else:
            r.append(0)

    hr = 1. if sum(r) > 0 else 0.

    # NDCG
    ndcg = math.log(2) / math.log(position + 2) if position < _K else 0

    # PRECISION (P)
    prec = sum(r) / len(r)

    # RECALL (R)
    rec = sum(r) / len(pos_predict)

    return hr, prec, rec, auc, ndcg


class Evaluator:
    def __init__(self, model, data, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.data = data
        self.k = k

        if data.validation_list:
            self.eval_feed_dicts_validation, self.eval_feed_dicts_test = _init_eval_model(data, val=True)
        else:
            self.eval_feed_dicts_test = _init_eval_model(data)

        self.model = model

    def eval(self, epoch=0, results={}, epoch_text='', start_time=0):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dict_test
        global _feed_dict_validation
        _dataset = self.data
        _model = self.model
        _K = self.k

        if self.data.validation_list:
            _feed_dict_validation, _feed_dict_test = self.eval_feed_dicts_validation, self.eval_feed_dicts_test
        else:
            _feed_dict_test = self.eval_feed_dicts_test

        res_test = []
        res_val = []

        eval_start_time = time()
        # all_predictions = self.model.predict_all().numpy()
        all_predictions, _ = self.model.predict_all()

        hr_v, prec_v, rec_v, auc_v, ndcg_v = '0', '0', '0', '0', '0'

        for user in range(self.model.data.num_users):
            current_prediction = all_predictions[user, :]
            if self.data.validation_list:
                res_test.append(_eval_by_user(user, current_prediction))
                res_val.append(_eval_by_user(user, current_prediction, val=True))
            else:
                res_test.append(_eval_by_user(user, current_prediction))

        res_test = list(filter(None, res_test))
        hr_t, prec_t, rec_t, auc_t, ndcg_t = (np.array(res_test).mean(axis=0)).tolist()
        if self.data.validation_list:
            res_val = list(filter(None, res_val))
            hr_v, prec_v, rec_v, auc_v, ndcg_v = (np.array(res_val).mean(axis=0)).tolist()
        print_results = \
            "%s \tTrain Time: %s \tEvaluation Time: %s" \
            "\nMetrics@%d (Validation)\n\t\tHR\tPrec\tRec\tAUC\tnDCG\n\t\t%f\t%f\t%f\t%f\t%f" \
            "\nMetrics@%d (Test)\n\t\tHR\tPrec\tRec\tAUC\tnDCG\n\t\t%f\t%f\t%f\t%f\t%f\n" % (
                epoch_text,
                datetime.timedelta(seconds=(time() - start_time)),
                datetime.timedelta(seconds=(time() - eval_start_time)),
                _K,
                hr_v,
                prec_v,
                rec_v,
                auc_v,
                ndcg_v,
                _K,
                hr_t,
                prec_t,
                rec_t,
                auc_t,
                ndcg_t
            )

        print(print_results)

        if len(epoch_text) != '':
            results[epoch] = {
                'hr_v': hr_v, 'auc_v': auc_v, 'p_v': prec_v, 'r_v': rec_v, 'ndcg_v': ndcg_v,
                'hr_t': hr_t, 'auc_t': auc_v, 'p_t': prec_t, 'r_t': rec_t, 'ndcg_t': ndcg_t
            }

        return print_results

    def store_recommendation(self, path=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (anonymized)
        attack_name: The name for the attack stored file
        :return:
        """
        # results = self.model.predict_all().numpy()
        results = np.array(self.model.predict_all())
        with open(path, 'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.training_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')

    def store_recommendation_attention(self, path=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (anonymized)
        attack_name: The name for the attack stored file
        :return:
        """
        # results = self.model.predict_all().numpy()
        results, attentions = self.model.predict_all()
        with open(path, 'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.training_list[u]] = -np.inf
                top_k_id = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k_id]
                for i, value in enumerate(top_k_id):
                    out.write(
                        str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\t' + \
                        str(attentions[u, value, 0]) + '\t' + str(attentions[u, value, 1]) + '\n'
                    )
