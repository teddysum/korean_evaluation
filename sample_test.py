import json
from datetime import datetime
from evaluation import evaluation


def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j


# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


if __name__ == '__main__':


    # test_file_path = './sample_classification_f1_test.jsonl'
    # submit_file_path = './sample_classification_f1_pred.jsonl'

    # test_file_path = './sample_mse_test.jsonl'
    # submit_file_path = './sample_mse_pred.jsonl'

    # test_file_path = '(정답지)nikluge-2022-nli-test-answer.jsonl'
    # submit_file_path = 'mse_test_236(100).jsonl'

    # test_file_path = 'sample_sa_test.jsonl'
    # submit_file_path = 'sample_sa_pred.jsonl'

    test_file_path = 'sample_generation_test.jsonl'
    submit_file_path = 'sample_generation_pred.jsonl'

    # test_file_path = './data/nikluge-sc-2023-test-answer.jsonl'
    # submit_file_path = '02.jsonl'

    # test_file_path = 'sample_multi-label-dict_true.jsonl'
    # submit_file_path = 'sample_multi-label-dict_pred.jsonl'

    log_file_path = 'log/'
    log_dict = {
        'evaluation_complete':{

        },
        'evalutation_fail':{

        }
    }
    # test file load
    test_data = jsonlload(test_file_path)

    # 제출파일 load, json, jsonl 두 형태 모두 처리
    try:
        submit_data = jsonload(submit_file_path)
    except:
        try:
            submit_data = jsonlload(submit_file_path)
        except:
            print(submit_file_path + ' 파일 형식 오류 - json, 또는 jsonl 형식이 아님')
            log_dict['evalutation_fail'][submit_file_path] = '파일 형식 오류 - json, 또는 jsonl 형식이 아님'

    # result = evaluation(submit_data, test_data, evaluation_metrics=['classification_micro_F1', 'classification_macro_F1', 'classification_weighted_F1'])
    # result = evaluation(submit_data, test_data, evaluation_metrics=['MSE'])
    # result = evaluation(submit_data, test_data, evaluation_metrics=['ROUGE-1', 'BLEU'], ratio=1, iteration=1)
    # result = evaluation(submit_data, test_data, evaluation_metrics=['sa_f1'], ratio=1, iteration=1)
    result = evaluation(submit_data, test_data, evaluation_metrics=['bleurt', 'bertscore', 'ROUGE-1', 'BLEU'])
    # result = evaluation(submit_data, test_data, evaluation_metrics=['bertscore', 'ROUGE-1', 'BLEU'])
    # result = evaluation(submit_data, test_data, evaluation_metrics=['multi_label_classification_micro_F1'])

    try:
        log_dict['evaluation_complete'][submit_file_path] = result
    except:
        print(submit_file_path + ' 파일 형식 오류 - 데이터 형태가 기준과 다름')
        log_dict['evalutation_fail'][submit_file_path] = '파일 형식 오류 - 데이터 형태가 제출 기준과 다름'

    # 로그파일 저장
    now = datetime.now()
    log_file_name = now.strftime('%Y-%m-%d_%H-%M-%S') + '.json'
    jsondump(log_dict, log_file_path + log_file_name)
    print(log_dict)
