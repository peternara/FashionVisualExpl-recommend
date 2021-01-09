import argparse
import pandas as pd
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Run logs to excel.")
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default="attentive_fashion", help="set recommendation model")
    parser.add_argument('--custom_match', nargs='?', default='', help='custom string to match')
    parser.add_argument('--param_to_sort', nargs='+', type=str, default=['lr', 'emk'],
                        help='list of parameters to sort on')
    parser.add_argument('--metrics', nargs='+', type=str, default=['hr', 'prec', 'rec', 'auc', 'ndcg'],
                        help='list of evaluated metrics')

    return parser.parse_args()


def logs_to_excel():
    args = parse_args()
    all_logs = glob.glob('../logs/' + args.rec + '-' + args.dataset + '*' + args.custom_match + '*')
    df = pd.DataFrame([], columns=list(args.param_to_sort) + [m + '_v' for m in list(args.metrics)] +
                                  [m + '_t' for m in list(args.metrics)])

    for log in all_logs:
        with open(log, 'r') as f:
            content = f.readlines()
        test_res = content[-7].split('\t')[2:]
        validation_res = content[-10].split('\t')[2:]
        test_res[-1] = test_res[-1][:-1]
        validation_res[-1] = validation_res[-1][:-1]
        test_res = [float(t) for t in test_res]
        validation_res = [float(v) for v in validation_res]
        current_list = []
        for s in log.split('-'):
            for a in args.param_to_sort:
                if a in s:
                    current_list.append(float(s.split(a)[1]) if ('.' in s.split(a)[1]) else int(s.split(a)[1]))
        current_list += validation_res
        current_list += test_res
        serie = pd.Series(current_list, index=df.columns)
        df = df.append(serie, ignore_index=True)

    df = df.groupby(list(args.param_to_sort)[-1]).apply(
        lambda x: x.sort_values(by=[list(args.param_to_sort)[0]], ascending=True)
    ).reset_index(drop=True)
    df.to_csv(
        '../logs/' + args.rec + '_' + args.dataset + ('_' + args.custom_match if args.custom_match else '') + '.tsv',
        sep='\t', header=False, index=False)


if __name__ == '__main__':
    logs_to_excel()
