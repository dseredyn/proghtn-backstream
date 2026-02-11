# Copyright (c) 2026 Dawid Seredyński

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict
from typing import Mapping, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

all_names_sorted =\
    ['c', 'e', 'a', 'f', 'i', 'j', 'h', 'g', 'l', 'm', 'p', 'k', 'y', 'z', 'w', 'o', 'x', 'y2', 'u', 'y3']

renamed_problems = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
renamed_map = {old_name: new_name for old_name, new_name in zip(all_names_sorted, renamed_problems)}
renamed_map['pddlstream'] = 'Z'

def load_jsons_one_level(root_dir: str | Path, filename: str, encoding: str = "utf-8"
                         ) -> Dict[Path, dict[str, Any]]:
    """
    Reads json file in each subdirectory of root_dir
    Returns: {file_path: json_data}
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)

    out: Dict[Path, dict[str, Any]] = {}

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        json_path = subdir / filename
        if not json_path.is_file():
            continue

        with json_path.open("r", encoding=encoding) as f:
            out[json_path] = json.load(f)

    return out


def plot_samples(
    all_names_sorted: list[str],
    samples: Mapping[str, Sequence[float]],
    success_rate_map: Mapping[str, str]|None,
    *,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_means: bool = True,
    show_medians: bool = True,
    show_extrema: bool = False,
    show_points: bool = True,
    jitter: float = 0.08,
    point_size: float = 18.0,
    empty_label: str = "no data",
    ax: Optional[plt.Axes] = None,
    y_lim: None|tuple[float, float] = None,
    use_box_plot = True,
) -> plt.Axes:
    """
    Violin plot dla wielu zmiennych. Jeśli lista próbek jest pusta,
    funkcja zaznaczy to tekstem na wykresie zamiast rzucać wyjątek.
    """

    # rename samples
    print(all_names_sorted)
    all_names_sorted_ren = [renamed_map[problem_id] for problem_id in all_names_sorted]
    all_names_sorted = all_names_sorted_ren
    samples_ren = {}
    if not success_rate_map is None:
        success_rate_map_ren = {}
    else:
        success_rate_map_ren = None
    for problem_id, sample in samples.items():
        new_problem_id = renamed_map[problem_id]
        samples_ren[new_problem_id] = sample
        if not success_rate_map_ren is None:
            assert not success_rate_map is None
            success_rate_map_ren[new_problem_id] = success_rate_map[problem_id]
    samples = samples_ren
    if not success_rate_map_ren is None:
        success_rate_map = success_rate_map_ren

    names = [name for name in all_names_sorted if name in samples]
    arrays = [np.asarray(samples[n], dtype=float) for n in names]

    if len(arrays) == 0:
        raise ValueError("samples is empty (no variables).")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    # Pozycje osi X dla wszystkich zmiennych (także pustych)
    positions_all = np.arange(1, len(names) + 1)

    # Tylko niepuste dane idą do violinplot
    nonempty = [(i, arr) for i, arr in enumerate(arrays) if arr.size > 0]
    if nonempty:
        idxs, data_nonempty = zip(*nonempty)
        pos_nonempty = [positions_all[i] for i in idxs]

        if use_box_plot:
            ax.boxplot(
                data_nonempty,
                positions=pos_nonempty,
                # showmeans=show_means,
                # showmedians=show_medians,
                # showextrema=show_extrema,
            )

        # Punkty próbek
        if show_points:
            rng = np.random.default_rng(0)
            for x, arr in zip(pos_nonempty, data_nonempty):
                if use_box_plot:
                    xj = x + rng.uniform(-jitter, jitter, size=arr.size)
                    ax.scatter(xj, arr, s=point_size)
                else:
                    xj = x
                    ax.bar(xj, arr)

    # Ustaw etykiety osi X (wszystkie, również puste)
    ax.set_xticks(positions_all)
    ax.set_xticklabels(names)

    if y_lim is None:
        # Print samples count
        y0, y1 = ax.get_ylim()

        # DODAJ: dodatkowy margines u góry
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            pad = 0.12 * (y1 - y0)     # np. +12% zakresu
            ax.set_ylim(y0, y1 + pad)
    else:
        ax.set_ylim(y_lim[0], y_lim[1])
    y0, y1 = ax.get_ylim()

    if not success_rate_map is None:
        # policz pozycję tekstu blisko góry
        y_text = y1 - 0.03 * (y1 - y0)

        for i, arr in enumerate(arrays):
            x = positions_all[i]
            success_rate = success_rate_map[names[i]]
            ax.text(x, y_text, success_rate, ha="center", va="top")

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlim(0.5, len(names) + 0.5)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    #figure(figsize=(8, 6), dpi=80)

    return ax


def summarizePDDLStreamTest(plot_out_dir: str):
    # https://github.com/caelan/pddlstream.git
    # problem A: python -m examples.pybullet.tamp.run -problem packed
    # problem B: 
    # problem C: 

    samples = {
        'pddlstream': [67.032, 86.760, 34.797, 48.505, 37.919, 29.445, 41.774, 31.129, 46.964, 32.603],
#        'B': [7.378, 25.602, 4.369, 20.324, 3.552, 25.421, 6.013, 7.336, 3.381, 3.337],
#        'C': [],
        }

    _, ax = plt.subplots(figsize=(1.5, 3))
    plot_samples(['pddlstream'], samples, None, ylabel="time [s]", y_lim=(0, 100), ax=ax)
    plt.tight_layout()
    plt.savefig(f'{plot_out_dir}/time_pddlstream.png')
    plt.savefig(f'{plot_out_dir}/time_pddlstream.pdf')

    samples = {
        'pddlstream': [32, 41, 20, 29, 20, 20, 20, 20, 29, 20],
#        'B': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
#        'C': [],
        }
    _, ax = plt.subplots(figsize=(1.5, 3))
    plot_samples(['pddlstream'], samples, None, ylabel="actions", y_lim=(0, 50), ax=ax)
    plt.tight_layout()
    plt.savefig(f'{plot_out_dir}/plan_pddlstream.png')
    plt.savefig(f'{plot_out_dir}/plan_pddlstream.pdf')


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        prog="planner",
        description="Run planner with a dynamically loaded plugin and config from tamp_htn_stream_examples.",
    )

    parser.add_argument(
        "--visualization-dir",
        help="Test suite .json file path: absolute OR package://<pkg-name>/<relative-path>",
    )
    args = parser.parse_args(argv)

    vis_path_base = args.visualization_dir
    plot_out_dir = f'{vis_path_base}/plot'

    try:
        os.makedirs(plot_out_dir)
    except:
        pass


    summarizePDDLStreamTest(plot_out_dir)

    # current_dir_list = [
    #     '2026-02-11']

    current_dir_list = []

    root = Path(vis_path_base)
    if not root.is_dir():
        raise NotADirectoryError(root)

    out: Dict[Path, dict[str, Any]] = {}

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == 'latest':
            continue
        current_dir_list.append(subdir)

    test_run_list = [
        ('backtrack, random_pdf', 'test_01_full', current_dir_list),
        ('backtrack, highest_score', 'test_01_gen_best', current_dir_list),
        ('backtrack, random', 'test_01_random', current_dir_list),
        ('no backtrack, random_pdf', 'test_01_no_backtrack', current_dir_list),
        ('no backtrack, highest_score', 'test_01_no_backtrack_best', current_dir_list),
    ]

    test_run_id_pretty = {test_run_id: pretty_name for pretty_name, test_run_id, _ in test_run_list}

    summary = []
    all_runs_problem_min_plan_length_map = {}
    for pretty_name, test_run_id, current_dir_list in test_run_list:
        #tests_dirs = [f'{vis_path_base}/{sub_dir}/' for sub_dir in current_dir_list]
        tests_dirs = current_dir_list

        sum_planning_time, plan_lengths, count_solved, count_total, problem_min_plan_length_map, sol_nodes_mean_factor = \
                                summarize_test_run(pretty_name, test_run_id, tests_dirs, plot_out_dir)
        summary.append( (test_run_id, sum_planning_time, plan_lengths, count_solved, count_total, sol_nodes_mean_factor) )

        for problem_id, min_len in problem_min_plan_length_map.items():
            if not problem_id in all_runs_problem_min_plan_length_map:
                all_runs_problem_min_plan_length_map[problem_id] = min_len
            else:
                all_runs_problem_min_plan_length_map[problem_id] = min(min_len, all_runs_problem_min_plan_length_map[problem_id])

    print(all_runs_problem_min_plan_length_map)
    # Minimum plan length for each problem for all tests
    samples = all_runs_problem_min_plan_length_map
    plot_samples(all_names_sorted, samples, None, title=f"Minimum plan length found", ylabel="actions",
                    y_lim=(0, 80), use_box_plot=False)
    plt.tight_layout()
    plt.savefig(f'{plot_out_dir}/plan.png')
    plt.savefig(f'{plot_out_dir}/plan.pdf')
    
    run_id_latex_map = {
        'backtrack, random':           '\\ablB{} & \\ablBacktrackOn{} & \\ablSamplingRnd{}',
        'backtrack, highest_score':    '\\ablA{} & \\ablBacktrackOn{} & \\ablSamplingOpt{}',
        'backtrack, random_pdf':       '\\ablBase{} & \\ablBacktrackOn{} & \\ablSamplingPdf{}',
        'no backtrack, random_pdf':    '\\ablC{} & \\ablBacktrackOff{} & \\ablSamplingPdf{}',
        'no backtrack, highest_score': '\\ablD{} & \\ablBacktrackOff{} & \\ablSamplingOpt{}'
    }
    # Print summary at the end
    print(f'name & time per action [s] & success rate [\\%] \\\\')
    for test_run_id, sum_succ_planning_time, plan_lengths, count_solved, count_total, sol_nodes_mean_factor in summary:
        tries_count = 10
        # For all unsolved tries, assume plan length as the minimum length, to better estimate
        # lower bound of action generation time
        sum_plan_length = 0
        #print(plan_lengths)
        for problem_id in plan_lengths:
            #print(f'problem_id: {problem_id}, plan_lengths: {plan_lengths}')
            sum_plan_length += sum( plan_lengths[problem_id] )
            #sum_plan_length += (tries_count - len(plan_lengths[problem_id])) * all_runs_problem_min_plan_length_map[problem_id]

        if sum_plan_length > 0:
            time_per_action = sum_succ_planning_time / sum_plan_length
        else:
            time_per_action = -1.0
        if count_total > 0:
            success_rate = float(count_solved) / count_total
        else:
            success_rate = -1.0
        pretty_name = test_run_id_pretty[test_run_id]

        # solution depth / total nodes
        # branches / total nodes
        #print(f'{run_id_latex_map[pretty_name]} & {time_per_action:.2f} & {100*success_rate:.1f}\\% \\\\  % {pretty_name}')
        print(f'{run_id_latex_map[pretty_name]} & {time_per_action:.2f} & '+\
              f'{100*success_rate:.1f} & {100.0*sol_nodes_mean_factor:.1f} \\\\')

    return 0


def getNodesCount(data):
    # count of all nodes:
    # 1 (root node) + decompositions + successful generations + executions
    decompositon_count = sum( [m['count'] for m_name, m in data['decompositions']['method'].items()] )
    succ_generations_count = sum( [s['success'] for s_name, s in data['streams'].items() if 'success' in s] )
    exec_count = sum( [a['count'] for a_class, a in data['executions'].items()] )
    result = 1 + decompositon_count + succ_generations_count + exec_count
    total_iterations = data['algorithm_run']['total_iterations']
    if result > total_iterations:
        raise Exception(f'result > total_iterations : {result} > {total_iterations}')
    return result

    # liczba dekompozycji  = suma ['decompositions']['method'][method_name]['count']
    # liczba udanych generacji = suma ['streams'][stream_name]['success']
    # liczba wykonań = ['executions'][action_class]['count']


def getSolutionDepth(data):
    if 'solutions' in data:
        for sol_node_id in data['solutions']:
            return data['solutions'][sol_node_id]['depth']
    return None


def summarize_test_run(pretty_name, test_run_id, tests_dirs, plot_out_dir):
    stats_files = {}
    for tests_dir in tests_dirs:
        for key, value in load_jsons_one_level(tests_dir, "stats.json").items():
            stats_files[key] = value

    print(f'summarize_test_run( {pretty_name}, {test_run_id})')

    problems_map = {}
    filenames_map = {}
    for filename, data in stats_files.items():
        if data['algorithm_run']['test_run_id'] != test_run_id:
            # Filter out other test runs
            continue
        # else:

        problem_id = data['algorithm_run']['problem_id']
        if not problem_id in problems_map:
            problems_map[problem_id] = []
            filenames_map[problem_id] = []
        problems_map[problem_id].append(data)
        filenames_map[problem_id].append(filename)

    # print(f'problems_map: {problems_map}')
    samples_map = {}
    for problem_id in sorted(problems_map.keys()):
        samples_map[problem_id] = {
        'iterations': [],
        'total_time': [],
        'max_node_depth': [],
        'plan_length': [],
        'solved_planning_time': [],
        'solved_or_not_planning_time': [],
        'planning_tries': 0,
        'nodes': [],
        'solution_depths': [],
        }

    # Statistics for streams across all problems
    streams_map = {}

    success_rate_map = {}
    for problem_id in sorted(problems_map.keys()):
        data_list = problems_map[problem_id]
        filenames_list = filenames_map[problem_id]
        count = len(data_list)
        solved_count = 0

        mean_iterations = 0
        min_iterations = 1000000
        max_iterations = 0
        mean_time = 0
        mean_max_node_depth = 0
        mean_plan_length = 0
        mean_plan_length_count = 0
        for filename, data in zip(filenames_list, data_list):
            has_solution = ('solutions' in data and len(data['solutions']) > 0)
            samples_map[problem_id]['planning_tries'] += 1
            total_time = data['algorithm_run']['total_time']
            samples_map[problem_id]['solved_or_not_planning_time'].append( total_time )

            samples_map[problem_id]['nodes'].append( getNodesCount(data) )
            samples_map[problem_id]['solution_depths'].append( getSolutionDepth(data) )

            for stream_name, stream in data['streams'].items():
                if not stream_name in streams_map:
                    streams_map[stream_name] = {
                        'sum_time': 0.0,
                        'sum_total_count': 0,
                        'sum_success': 0,
                        'sum_bad_param_gen': 0,
                        'backtracks': 0,
                    }
                if not 'total_count' in stream:
                    continue
                streams_map[stream_name]['sum_time'] += stream['time']
                streams_map[stream_name]['sum_total_count'] += stream['total_count']
                if 'success' in stream:
                    streams_map[stream_name]['sum_success'] += stream['success']
                if 'bad_param_gen' in stream:
                    streams_map[stream_name]['sum_bad_param_gen'] += stream['bad_param_gen']
                if 'backtracks' in stream:
                    streams_map[stream_name]['backtracks'] += stream['backtracks']

            if not has_solution:
                print(f'Missing solution for {filename}')
                continue

            solved_count += 1
            iterations = data['algorithm_run']['total_iterations']
            max_node_depth = data['algorithm_run']['max_node_depth']
            min_iterations = min(min_iterations, iterations)
            mean_iterations += iterations
            max_iterations = min(max_iterations, iterations)

            mean_time += total_time
            mean_max_node_depth += total_time
            samples_map[problem_id]['iterations'].append( iterations )
            samples_map[problem_id]['total_time'].append( total_time )
            samples_map[problem_id]['max_node_depth'].append( max_node_depth )

            if has_solution:
                for node_id in data['solutions']:
                    if 'plan_length' in data['solutions'][node_id]:
                        plan_length = data['solutions'][node_id]['plan_length']
                        samples_map[problem_id]['plan_length'].append( plan_length )
                        samples_map[problem_id]['solved_planning_time'].append( total_time )
                        mean_plan_length += plan_length
                        mean_plan_length_count += 1

        mean_iterations = mean_iterations / count
        mean_time = mean_time / count
        mean_max_node_depth = mean_max_node_depth / count
        if mean_plan_length_count > 0:
            mean_plan_length = mean_plan_length / mean_plan_length_count
        else:
            mean_plan_length = None

        # success_rate_map[problem_id] = f'{solved_count}\n{count}'
        # success_rate_map[problem_id] = f'{count}\n{solved_count/count:.1f}'
        if solved_count == count:
            success_rate_map[problem_id] = f'1'
        else:
            success_rate_map[problem_id] = f'{solved_count/count:.1f}'
        #success_rate_map[problem_id] += f'\n{count}'
        # success_rate_map[problem_id] = f'{solved_count}'
        print(f'problem {problem_id}, sol: {solved_count}/{count}, it: {mean_iterations:.1f}, '+\
              f'time: {mean_time:.1f}, depth: {mean_max_node_depth:.1f}, plan: {mean_plan_length}')

        # data['algorithm_run']['max_iterations']
        # data['algorithm_run']['generator_sampling_type']
        # data['algorithm_run']['use_backtracking']
        # data['algorithm_run']['repeat_idx']
        # data['algorithm_run']['test_run_id']

    # samples = {
    #     "X": [0.9, 1.1, 1.0, 1.2, 0.8, 1.05, 0.95, 1.15, 1.0, 0.85],
    #     "Y": [2.0, 2.1, 1.9, 2.2, 2.05, 2.15, 1.95, 2.3, 2.05, 2.1],
    #     "Z": [0.2, 0.25, 0.22, 0.3, 0.18, 0.27, 0.21, 0.24, 0.26, 0.19],
    # }

    if test_run_id == 'test_01_full':
        # For each problem calculate mean planning time to determine difficulty
        problem_mean_time = []
        for problem_id in samples_map:
            mean_time = sum([x for x in samples_map[problem_id]['total_time']]) / len(samples_map[problem_id]['total_time'])
            problem_mean_time.append( (mean_time, problem_id) )
        print(f'Sorted problems by mean planning time:')
        sorted_problem_mean_time = sorted(problem_mean_time, key=lambda x:x[0])
        for mean_time, problem_id in sorted_problem_mean_time:
            print(f'  {problem_id}, {mean_time}')
        print(f'{[x[1] for x in sorted_problem_mean_time]}')

        # Summarize generators
        # Table of generators (sorted from the most time-consuming)
        # name, % count, % time, mean time [s], success rate, % blames in all backtracks
        gen_summary = []
        all_problems_planning_time = sum([sum(sample['total_time']) for problem_id, sample in samples_map.items()])
        all_generations_count = sum([str_info['sum_total_count'] for str_name, str_info in streams_map.items()])
        all_backtracks_count = sum([str_info['backtracks'] for str_name, str_info in streams_map.items() if 'backtracks' in str_info])
        all_blames_count = sum([str_info['sum_bad_param_gen'] for str_name, str_info in streams_map.items() if 'sum_bad_param_gen' in str_info])

        total_gen_time = 0.0
        all_successful_gen = 0
        all_bad_param_gen = 0
        for stream_name, stream_info in streams_map.items():
            total_gen_time += stream_info['sum_time']            

            percent_time = 100 * stream_info['sum_time'] / all_problems_planning_time
            mean_time = stream_info['sum_time'] / stream_info['sum_total_count']

            percent_count = 100.0 * stream_info['sum_total_count'] / all_generations_count

            success_rate = 100.0 * stream_info['sum_success'] / stream_info['sum_total_count']
            all_successful_gen += stream_info['sum_success']
            
            gen_count = stream_info['sum_total_count']
            # if all_backtracks_count == 0:
            #     blame_factor = 0
            # else:
            blame_factor = 100.0 * stream_info['sum_bad_param_gen'] / all_blames_count
            all_bad_param_gen += stream_info['sum_bad_param_gen']

            # gen_summary.append( [stream_name, f'{percent_count:.1f}', f'{percent_time:.1f}',
            #         f'{mean_time:.1f}', f'{success_rate:.1f}', f'{blame_factor:.1f}'] )
            gen_summary.append( [stream_name, percent_count, percent_time,
                    mean_time, gen_count, success_rate, blame_factor] )
        
        gen_summary = sorted(gen_summary, key=lambda x: x[2], reverse=True)

        # The last row is all streams summary
        all_gen_time_percent = 100.0 * total_gen_time / all_problems_planning_time
        all_gen_mean_time = total_gen_time / all_generations_count
        total_success_rate = 100.0 * all_successful_gen / all_generations_count

        # if all_backtracks_count == 0:
        #     all_blame_factor = 0
        # else:
        all_blame_factor = 100.0 * all_bad_param_gen / all_blames_count
        
        gen_summary.append( ['all', 100.0, all_gen_time_percent, all_gen_mean_time,
                             all_generations_count, total_success_rate, all_blame_factor] )

        exclude_gen = [
            'GenGraspReservedSpace',
            'GenGraspTrajRev',
            'GenPregraspHandConfig',
            'GenLookAt',
            'GenFk',
            'GenGraspHandConfig',
            'GenTrajRev',
            'GenCurrentConfG',
            'GenCurrentConfAT',
            'GenCurrentConfT',
            'GenCurrentConfH',
            'GenClosedG',
            'GenManipSide',
            'GenResourceId',
            'GenOtherSide',
            'GenAnySidePref',
            'GenFkG',
            'GenCurrentGraspAtSide',
        ]

        txt = ''
        for idx, (stream_name, percent_count,\
                  percent_time, mean_time, gen_count, success_rate, blame_factor) in enumerate(gen_summary):
            if stream_name in exclude_gen:
                stream_name = f'%{stream_name}'
            txt += f'{stream_name} & {percent_time:.1f} & '+\
                   f'{mean_time:.1f} & {percent_count:.1f} & {success_rate:.1f} & {blame_factor:.1f} \\\\ \n'
            if idx == len(gen_summary) - 1:
                txt += '\\hline \n'
                txt += f'{stream_name} & {percent_time:.1f} & '+\
                        f'{mean_time:.1f} & -- & '+\
                        f'{success_rate:.1f} & -- \\\\ \n'

        print('******** generators summary **********')
        print(f'  all_backtracks_count: {all_backtracks_count}')
        print(f'  all generations count: {all_generations_count}')
        print(f'  all generations time: {total_gen_time}')
        print(txt)
        print(f'All generators planning time factor: '+\
              f'{100 * total_gen_time / all_problems_planning_time:.2f} %')

        mean_planning_time_problems = ['j', 'i', 'h', 'g', 'l', 'm', 'y', 'p']
        mean_planning_time = 0.0
        mean_planning_time_count = 0
        for problem_id in mean_planning_time_problems:
            if not problem_id in samples_map:
                continue
            mean_planning_time += sum(samples_map[problem_id]['solved_planning_time'])
            mean_planning_time_count += len(samples_map[problem_id]['solved_planning_time'])
        if mean_planning_time_count > 0:
            mean_planning_time = mean_planning_time / mean_planning_time_count
        else:
            mean_planning_time = 0.0
        print(f'Mean planning time for problems {mean_planning_time_problems} is {mean_planning_time:.1f}')

    

    for stream_name, str_info in streams_map.items():
        stream_name_lj = stream_name.ljust(25)
        if str_info['sum_time'] < 1.0:
            continue
        print(f'{stream_name_lj} sum_time: {str_info['sum_time']:.1f}, '+\
              f'success_rate: {str_info['sum_success']}/{str_info['sum_total_count']}, '+\
              f'bad gen: {str_info['sum_bad_param_gen']}')
        # str_info['sum_time']
        # str_info['sum_total_count']
        # str_info['sum_success']
        # str_info['sum_bad_param_gen'] 
    out_filenames_map = {
        'test_01_full': 'base',
        'test_01_gen_best': 'b-on_s-opt',
        'test_01_random': 'b-on_s-rnd',
        'test_01_no_backtrack': 'b-off_s-pdf',
        'test_01_no_backtrack_best': 'b-off_s-opt',
    }
    filename = out_filenames_map[test_run_id]

    # Planning time for each problem
    samples = {problem_id: values['total_time'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        # plot_samples(all_names_sorted, samples, success_rate_map, title=f"Total planning time", ylabel="time [s]",
        #              y_lim=(0, 550))
        plot_samples(all_names_sorted, samples, success_rate_map, ylabel="time [s]", y_lim=(0, 550))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/time_{filename}.png')
        plt.savefig(f'{plot_out_dir}/time_{filename}.pdf')
        # plt.show()

    # Number of iterations for each problem
    samples = {problem_id: values['iterations'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        plot_samples(all_names_sorted, samples, success_rate_map, title=f"Iterations", ylabel="iterations",
                     y_lim=(0, 2000))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/iterations_{filename}.png')
        plt.savefig(f'{plot_out_dir}/iterations_{filename}.pdf')

    # samples = {problem_id: values['max_node_depth'] for problem_id, values in samples_map.items()}
    # plot_samples(all_names_sorted, samples, success_rate_map, title="Maximum depth of the search tree", ylabel="depth")
    # plt.tight_layout()
    # plt.show()

    # Plan length for each problem
    samples = {problem_id: values['plan_length'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        plot_samples(all_names_sorted, samples, None, ylabel="actions", y_lim=(0, 80))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/plan_{filename}.png')
        plt.savefig(f'{plot_out_dir}/plan_{filename}.pdf')


    # print(f'samples_map: {samples_map}')
    # Planning time per one action in the plan for each algorithm variant.
    # Calculate for solved problems only
    sum_succ_planning_time = 0.0
    #sum_plan_length = 0
    plan_lengths = {}
    count_solved = 0
    count_total = 0
    problem_min_plan_length_map = {}

    sol_nodes_mean_factor = 0.0
    sol_nodes_mean_factor_count = 0
    for problem_id, values in samples_map.items():
        count_solved += len(values['solved_planning_time'])
        count_total += values['planning_tries']
        #sum_plan_length += sum( [plan_length for plan_length in values['plan_length']] )
        plan_lengths[problem_id] = values['plan_length']
        # sum_planning_time += sum( [planning_time for planning_time in values['solved_or_not_planning_time']] )
        sum_succ_planning_time += sum( [planning_time for planning_time in values['solved_planning_time']] )

        if 'plan_length' in samples_map[problem_id] and len(samples_map[problem_id]['plan_length']) > 0:
            problem_min_plan_length_map[problem_id] = min(samples_map[problem_id]['plan_length'])

        for sol_node_depth, all_nodes in zip(samples_map[problem_id]['solution_depths'], samples_map[problem_id]['nodes']):
            if not sol_node_depth is None:
                sol_nodes_mean_factor += sol_node_depth / all_nodes
                sol_nodes_mean_factor_count += 1

    if sol_nodes_mean_factor_count > 0:
        sol_nodes_mean_factor = sol_nodes_mean_factor / sol_nodes_mean_factor_count
    else:
        sol_nodes_mean_factor = 0.0
#    return (sum_planning_time, sum_plan_length, count_solved, count_total, problem_min_plan_length_map)
    return (sum_succ_planning_time, plan_lengths, count_solved, count_total, problem_min_plan_length_map, sol_nodes_mean_factor)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
