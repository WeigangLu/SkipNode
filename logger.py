import numpy as np
import os


class Logger(object):
    def __init__(self, args):
        self.results = [0 for _ in range(args.runs)]
        self.run_times = [0 for _ in range(args.runs)]
        self.args = args
        self.folder_name = f"./results/{args.setting}/{args.dataset}"

    def name_change(self, dataset):
        self.folder_name = f"./results/{self.args.setting}/{dataset}"
    def add_result(self, run, result, res_type="acc"):
        assert 0 <= run < len(self.results)
        if res_type == "acc":
            self.results[run] = result
        elif res_type == "time":
            self.run_times[run] = result


    def dump_parameters(self, content):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        strategy_name = self.args.strategy
        if strategy_name == "SkipNode":
            strategy_name += self.args.skip_node_type
        filename = os.path.join(self.folder_name, self.args.model + "_{}.txt".format(strategy_name))
        with open(filename, "a+") as f:
            f.write(content + "\t")
            f.write(str(self.args) + "\n")

    def print_statistics(self):
        acc_mean = np.mean(self.results)
        acc_std = np.std(self.results)
        time_mean = np.mean(self.run_times)
        content = f'Final Test: {acc_mean:.2f} ± {acc_std:.2f} | Run Time: {time_mean:.5f}'
        print(content)
        self.dump_parameters(content)
