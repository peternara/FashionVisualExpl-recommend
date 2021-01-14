import argparse
import pandas as pd
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Run logs to excel.")
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset name')
    parser.add_argument('--rec', nargs='?', default="bprmf", help="set recommendation model")
    parser.add_argument('--param_to_sort', nargs='+', type=str, default=['lr', 'emk', 'reg'],
                        help='list of parameters to sort on')
    parser.add_argument('--regs', nargs='+', type=float, default=[0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                        help='list of regularization values')
    parser.add_argument('--metrics', nargs='+', type=str, default=['hr', 'prec', 'rec', 'auc', 'ndcg'],
                        help='list of evaluated metrics')

    return parser.parse_args()


def logs_to_excel():
    args = parse_args()
    all_logs = glob.glob('../logs/@20/' + args.rec + '-' + args.dataset + '*')
    df = pd.DataFrame([], columns=list(args.param_to_sort) + [m + '_v' for m in list(args.metrics)] +
                                  [m + '_t' for m in list(args.metrics)])

    for log in all_logs:
        with open(log, 'r') as f:
            log_content = f.readlines()

        content = []
        reg_index = 0
        for num_line, line in enumerate(log_content):
            if line == 'END REGULARIZATION\n':
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
                current_list += [args.regs[reg_index]]
                current_list += validation_res
                current_list += test_res
                serie = pd.Series(current_list, index=df.columns)
                df = df.append(serie, ignore_index=True)
                content = []
                reg_index += 1
            else:
                content.append(line)

    df = df.groupby(list(args.param_to_sort)[-2]).apply(
        lambda x: x.sort_values(by=[list(args.param_to_sort)[0]],
                                ascending=True).groupby(list(args.param_to_sort)[0]).apply(
            lambda y: y.sort_values(by=[list(args.param_to_sort)[-1]])
        )
    ).reset_index(drop=True)
    df.to_csv(
        '../logs/' + args.rec + '_' + args.dataset + ('_' + args.custom_match if args.custom_match else '') + '.tsv',
        sep='\t', header=False, index=False)


if __name__ == '__main__':
    logs_to_excel()
