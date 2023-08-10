from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from nltk.translate.bleu_score import sentence_bleu
from rouge_metric import Rouge
import random
from konlpy.tag import Mecab
from bleurt import score
import evaluate


bert_scorer = evaluate.load('bertscore')
bert_model_type = 'bert-base-multilingual-cased'

checkpoint = "BLEURT-20"

scorer = score.BleurtScorer(checkpoint)

tokenizer = Mecab()


def data_sampling(true_data_list, pred_data_list, ratio):
    index_list = list(range(len(true_data_list)))
    random.shuffle(index_list)

    sampled_true_data_list = [0 for _ in range(len(true_data_list))]
    sampled_pred_data_list = [0 for _ in range(len(pred_data_list))]
    for i in range(len(true_data_list)):
        sampled_true_data_list[i] = true_data_list[index_list[i]]
        sampled_pred_data_list[i] = pred_data_list[index_list[i]]

    ratio_idx = int(len(true_data_list) * ratio)

    return sampled_true_data_list[:ratio_idx], sampled_pred_data_list[:ratio_idx]

def calc_multi_label_classification_micro_F1(true, pred):

    if type(true[0]) is list:
        if type(true[0][0]) is int:
            pass
        elif type(true[0][0]) is float:
            pass
        elif type(true[0][0]) is bool:
            pass
        elif type(true[0][0]) is str:
            pass
        else:
            return -1

    elif type(true[0]) is dict:

        sample_key = next(iter(true[0]))

        if type(true[0][sample_key]) is int:
            pass
        elif type(true[0][sample_key]) is float:
            pass
        elif type(true[0][sample_key]) is str:
            def dict_to_list(input_dict):
                output_list = []
                for instance in input_dict.values():
                    if instance == 'True' or instance == 'true':
                        output_list.append(1)
                    else:
                        output_list.append(0)

                return output_list

            formated_pred = list(map(lambda x: dict_to_list(x), pred))
            formated_true = list(map(lambda x: dict_to_list(x), true))
            f1_micro = f1_score(y_true=formated_true, y_pred=formated_pred, average='micro')

            return f1_micro

        elif type(true[0][sample_key]) is bool:
            def dict_to_list(input_dict):
                output_list = []
                for instance in input_dict.values():
                    if instance is True:
                        output_list.append(1)
                    else:
                        output_list.append(0)

            formated_pred = list(map(lambda x: dict_to_list(x), pred))
            formated_true = list(map(lambda x: dict_to_list(x), true))
            f1_micro = f1_score(y_true=formated_true, y_pred=formated_pred, average='micro')
            return f1_micro

        else:
            return -1
    else:
        return -1


def calc_classification_micro_F1(true, pred):
    return f1_score(true, pred, average='micro')


def calc_classification_macro_F1(true, pred):
    return f1_score(true, pred, average='macro')


def calc_classification_weighted_F1(true, pred):
    return f1_score(true, pred, average='weighted')


def calc_MSE(true, pred):
    for i in range(len(true)):
        if type(true[i]) == str:
            true[i] = float(true[i])
        if type(pred[i]) == str:
            pred[i] = float(pred[i])
    return mean_squared_error(true, pred)


def calc_ROUGE_1(true, pred):
    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.0,
    )

    scores = rouge_evaluator.get_scores(pred, true)
    return scores['rouge-1']['f']


def calc_BLEU(true, pred, apply_avg=True, apply_best=False, use_mecab=True):
    stacked_bleu = []

    if type(true[0]) is str:
        true = list(map(lambda x: [x], true))

    for i in range(len(true)):
        best_bleu = 0
        sum_bleu = 0
        for j in range(len(true[i])):

            if use_mecab:
                ref = tokenizer.morphs(true[i][j])
                candi = tokenizer.morphs(pred[i])
            else:
                ref = true[i][j].split()
                candi = pred[i].split()


            score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))

            sum_bleu += score
            if score > best_bleu:
                best_bleu = score

        avg_bleu = sum_bleu / len(true[i])
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

    return sum(stacked_bleu) / len(stacked_bleu)


def evaluation_sa_f1(true_data, pred_data):
    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):

        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano in true_data_list[i]:
            y_category = y_ano[0]
            y_polarity = y_ano[2]
            for p_ano in pred_data_list[i]:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano in true_data_list[i]:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

            is_ce_found = False
            is_pipeline_found = False

    # ce_precision = ce_eval['TP']/(ce_eval['TP']+ce_eval['FP'])
    # ce_recall = ce_eval['TP']/(ce_eval['TP']+ce_eval['FN'])
    #
    # ce_result = {
    #     'Precision': ce_precision,
    #     'Recall': ce_recall,
    #     'F1': 2*ce_recall*ce_precision/(ce_recall+ce_precision)
    # }

    pipeline_precision = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FP'])
    pipeline_recall = pipeline_eval['TP'] / (pipeline_eval['TP'] + pipeline_eval['FN'])
    if pipeline_recall == 0 and pipeline_precision == 0:
        pipeline_f1 = 0
    else:
        pipeline_f1 = 2 * pipeline_recall * pipeline_precision / (pipeline_recall + pipeline_precision)

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': pipeline_f1
    }

    return {
        "sa_f1": pipeline_result['F1']
    }
    # return {
    #     'category extraction result': ce_result,
    #     'entire pipeline result': pipeline_result
    # }


def calc_bleurt(true_data, pred_data):
    if type(true_data[0]) is list:
        true_data = list(map(lambda x: x[0], true_data))

    scores = scorer.score(references=true_data, candidates=pred_data)

    return sum(scores) / len(scores)


def calc_bertscore(true_data, pred_data):
    if type(true_data[0]) is list:
        true_data = list(map(lambda x: x[0], true_data))

    scores = bert_scorer.compute(predictions=pred_data, references=true_data, model_type=bert_model_type)

    return sum(scores['f1']) / len(scores['f1'])


def evaluation(inferenced_data, ground_truth, evaluation_metrics=[], ratio=1, iteration=1):
    temp_ground_truth_dict = {}
    true_data_list = []
    pred_data_list = []

    if len(inferenced_data) != len(ground_truth):
        return {
            'error': '제출 파일과 정답 파일의 데이터 개수가 서로 다름'
        }

    # sa_f1 인 경우
    if 'sa_f1' in evaluation_metrics:
        # 데이터 list로 변경
        for data in ground_truth:
            if data['id'] in temp_ground_truth_dict:
                return {
                    "error": "정답 데이터에 중복된 id를 가지는 경우 존재"
                }
            temp_ground_truth_dict[data['id']] = data['annotation']

        for data in inferenced_data:
            if data['id'] not in temp_ground_truth_dict:
                return {
                    "error": "제출 파일과 정답 파일의 id가 일치하지 않음"
                }
            true_data_list.append(temp_ground_truth_dict[data['id']])
            pred_data_list.append(data['annotation'])
        sampled_true_data_list, sampled_pred_data_list = data_sampling(true_data_list, pred_data_list, ratio)

        return evaluation_sa_f1(sampled_true_data_list, sampled_pred_data_list)

    # 평가 가능한 metric 목록
    defined_evaluation_metric_list = ['classification_micro_F1', 'classification_macro_F1',
                                      'classification_weighted_F1', 'MSE', 'ROUGE-1', 'BLEU', 'bleurt', 'bertscore', 'multi_label_classification_micro_F1']
    metric_to_func = {
        "classification_micro_F1": calc_classification_micro_F1,
        "classification_macro_F1": calc_classification_macro_F1,
        "classification_weighted_F1": calc_classification_weighted_F1,
        "MSE": calc_MSE,
        "ROUGE-1": calc_ROUGE_1,
        "BLEU": calc_BLEU,
        "bleurt": calc_bleurt,
        "bertscore": calc_bertscore,
        'multi_label_classification_micro_F1': calc_multi_label_classification_micro_F1
    }

    # 평가 대상 metric 정리
    for metric in evaluation_metrics:
        if metric not in defined_evaluation_metric_list:
            return {
                "error": f"evaluation metric 중 {metric}은 정의된 metric에 포함되어있지 않습니다."
            }

    evaluation_result = {metric: [] for metric in evaluation_metrics}

    # 데이터 list로 변경
    for data in ground_truth:
        if data['id'] in temp_ground_truth_dict:
            return {
                "error": "정답 데이터에 중복된 id를 가지는 경우 존재"
            }
        temp_ground_truth_dict[data['id']] = data['output']

    for data in inferenced_data:
        if data['id'] not in temp_ground_truth_dict:
            return {
                "error": "제출 파일과 정답 파일의 id가 일치하지 않음"
            }
        true_data_list.append(temp_ground_truth_dict[data['id']])
        pred_data_list.append(data['output'])

    # 평가 - iteration회 만금 ratio 비율로 sampling 한 데이터에 대해 평가
    for i in range(iteration):
        sampled_true_data_list, sampled_pred_data_list = data_sampling(true_data_list, pred_data_list, ratio)
        for metric in evaluation_metrics:
            evaluation_result[metric].append(metric_to_func[metric](sampled_true_data_list, sampled_pred_data_list))


    #iteration 만큼 반복된 결과에 대한 평균
    for key, value in evaluation_result.items():
        evaluation_result[key] = sum(value) / len(value)

    return evaluation_result
