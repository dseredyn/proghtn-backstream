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

import json
from pathlib import Path
from typing import Any, Dict
from typing import Mapping, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


all_names_sorted =\
    ['c', 'e', 'a', 'f', 'j', 'i', 'h', 'g', 'l', 'k', 'm', 'y', 'p', 'z', 'w', 'o', 'x', 'u', 'y2', 'y3']
    
def load_jsons_one_level(root_dir: str | Path, filename: str, encoding: str = "utf-8"
                         ) -> Dict[Path, dict[str, Any]]:
    """
    Reads json file in each subdirectory of root_dir
    Returns: {fiel_path: json_data}
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
    #names = list(samples.keys())
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

    # Zaznacz puste zmienne tekstem
    # empty_positions = [positions_all[i] for i, arr in enumerate(arrays) if arr.size == 0]
    # if empty_positions:
    #     # sensowna wysokość napisu: trochę ponad aktualny zakres osi Y,
    #     # a jeśli nie ma danych w ogóle, to użyj 0.0
    #     y0, y1 = ax.get_ylim()
    #     if not np.isfinite(y0) or not np.isfinite(y1) or (y0 == 0 and y1 == 1):
    #         # fallback (np. gdy wszystkie listy puste)
    #         y_text = 0.0
    #     else:
    #         y_text = y1 - 0.05 * (y1 - y0)

    #     for x in empty_positions:
    #         ax.text(x, y_text, empty_label, ha="center", va="top")

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


plot_out_dir = '/home/dseredyn/ws_tamp/visualization/plots'


def main(argv=None) -> int:

    base_dir = '/home/dseredyn/ws_tamp/visualization/planning',
    # current_dir_list = ['test_suite_all_11_tasks_8_runs', 'test_suite_all_11_tasks_8_runs_added',
    #                     'test_suite_all_two_runs_added']
    current_dir_list = [
        # '2026-02-05/old/test_01_full',
        # '2026-02-05/old/test_01_gen_best',
        # '2026-02-05/old/test_01_no_backtrack_best',
        # '2026-02-05/old/test_01_no_backtrack',
        # '2026-02-05/old/test_01_random',
        '2026-02-05/new/test_01_full',
        '2026-02-05/new/test_01_gen_best',
        '2026-02-05/new/test_01_no_backtrack_best',
        '2026-02-05/new/test_01_no_backtrack',
        '2026-02-05/new/test_01_random',
        '2026-02-05',
        '2026-02-06']
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
        tests_dirs = [f'/home/dseredyn/ws_tamp/visualization/planning/{sub_dir}/' for sub_dir in current_dir_list]

        sum_planning_time, plan_lengths, count_solved, count_total, problem_min_plan_length_map = \
                                summarize_test_run(pretty_name, test_run_id, tests_dirs)
        summary.append( (test_run_id, sum_planning_time, plan_lengths, count_solved, count_total) )

        for problem_id, min_len in problem_min_plan_length_map.items():
            if not problem_id in all_runs_problem_min_plan_length_map:
                all_runs_problem_min_plan_length_map[problem_id] = min_len
            else:
                all_runs_problem_min_plan_length_map[problem_id] = min(min_len, all_runs_problem_min_plan_length_map[problem_id])

    print(all_runs_problem_min_plan_length_map)
    # Minimum plan length for each problem for all tests
    samples = all_runs_problem_min_plan_length_map
    plot_samples(samples, None, title=f"Minimum plan length found", ylabel="actions",
                    y_lim=(0, 80), use_box_plot=False)
    plt.tight_layout()
    global plot_out_dir
    plt.savefig(f'{plot_out_dir}/png/plan.png')
    plt.savefig(f'{plot_out_dir}/pdf/plan.pdf')
    
    run_id_latex_map = {
        'backtrack, random':           '\\ablB{} & \\ablBacktrackOn{} & \\ablSamplingRnd{}',
        'backtrack, highest_score':    '\\ablA{} & \\ablBacktrackOn{} & \\ablSamplingOpt{}',
        'backtrack, random_pdf':       '\\ablBase{} & \\ablBacktrackOn{} & \\ablSamplingPdf{}',
        'no backtrack, random_pdf':    '\\ablC{} & \\ablBacktrackOff{} & \\ablSamplingPdf{}',
        'no backtrack, highest_score': '\\ablD{} & \\ablBacktrackOff{} & \\ablSamplingOpt{}'
    }
    # Print summary at the end
    print(f'name & time per action [s] & success rate [\\%] \\\\')
    for test_run_id, sum_succ_planning_time, plan_lengths, count_solved, count_total in summary:
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
        #print(f'{run_id_latex_map[pretty_name]} & {time_per_action:.2f} & {100*success_rate:.1f}\\% \\\\  % {pretty_name}')
        print(f'{run_id_latex_map[pretty_name]} & {time_per_action:.2f} & {100*success_rate:.1f} \\\\')

    return 0


def summarize_test_run(pretty_name, test_run_id, tests_dirs):
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
        success_rate_map[problem_id] += f'\n{count}'
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
            
            # if all_backtracks_count == 0:
            #     blame_factor = 0
            # else:
            blame_factor = 100.0 * stream_info['sum_bad_param_gen'] / all_blames_count
            all_bad_param_gen += stream_info['sum_bad_param_gen']

            # gen_summary.append( [stream_name, f'{percent_count:.1f}', f'{percent_time:.1f}',
            #         f'{mean_time:.1f}', f'{success_rate:.1f}', f'{blame_factor:.1f}'] )
            gen_summary.append( [stream_name, percent_count, percent_time,
                    mean_time, success_rate, blame_factor] )
        
        gen_summary = sorted(gen_summary, key=lambda x: x[2], reverse=True)

        # The last row is all streams summary
        all_gen_time_percent = 100.0 * total_gen_time / all_problems_planning_time
        all_gen_mean_time = all_gen_time_percent / all_generations_count
        total_success_rate = 100.0 * all_successful_gen / all_generations_count

        # if all_backtracks_count == 0:
        #     all_blame_factor = 0
        # else:
        all_blame_factor = 100.0 * all_bad_param_gen / all_blames_count
        
        gen_summary.append( ['all', 100.0, all_gen_time_percent, all_gen_mean_time,
                             total_success_rate, all_blame_factor] )

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
                  percent_time, mean_time, success_rate, blame_factor) in enumerate(gen_summary):
            if idx == len(gen_summary) - 1:
                txt += '\\hline \n'
            if stream_name in exclude_gen:
                stream_name = f'%{stream_name}'
            # txt += f'{stream_name} & {percent_count:.1f} & {percent_time:.1f} & '+\
            #         f'{mean_time:.1f} & {success_rate:.1f} & {blame_factor:.1f} \\\\ \n'
            txt += f'{stream_name} & {percent_time:.1f} & '+\
                    f'{mean_time:.1f} & {success_rate:.1f} & {blame_factor:.1f} \\\\ \n'

        print('******** generators summary **********')
        print(f'  all_backtracks_count: {all_backtracks_count}')
        print(txt)
        print(f'All generators planning time factor: '+\
              f'{100 * total_gen_time / all_problems_planning_time:.2f} %')

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

    global plot_out_dir
    # Planning time for each problem
    samples = {problem_id: values['total_time'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        # plot_samples(samples, success_rate_map, title=f"Total planning time", ylabel="time [s]",
        #              y_lim=(0, 550))
        plot_samples(samples, success_rate_map, ylabel="time [s]", y_lim=(0, 550))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/png/time_{filename}.png')
        plt.savefig(f'{plot_out_dir}/pdf/time_{filename}.pdf')
        # plt.show()

    # Number of iterations for each problem
    plot_out_dir = '/home/dseredyn/ws_tamp/visualization/plots'
    samples = {problem_id: values['iterations'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        plot_samples(samples, success_rate_map, title=f"Iterations", ylabel="iterations",
                     y_lim=(0, 2000))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/png/iterations_{filename}.png')
        plt.savefig(f'{plot_out_dir}/pdf/iterations_{filename}.pdf')

    # samples = {problem_id: values['max_node_depth'] for problem_id, values in samples_map.items()}
    # plot_samples(samples, success_rate_map, title="Maximum depth of the search tree", ylabel="depth")
    # plt.tight_layout()
    # plt.show()

    # Plan length for each problem
    samples = {problem_id: values['plan_length'] for problem_id, values in samples_map.items()}
    if len(samples) > 0:
        plot_samples(samples, success_rate_map, title=f"Plan length", ylabel="actions",
                     y_lim=(0, 80))
        plt.tight_layout()
        plt.savefig(f'{plot_out_dir}/png/plan_{filename}.png')
        plt.savefig(f'{plot_out_dir}/pdf/plan_{filename}.pdf')


    # print(f'samples_map: {samples_map}')
    # Planning time per one action in the plan for each algorithm variant.
    # Calculate for solved problems only
    sum_succ_planning_time = 0.0
    #sum_plan_length = 0
    plan_lengths = {}
    count_solved = 0
    count_total = 0
    problem_min_plan_length_map = {}
    for problem_id, values in samples_map.items():
        count_solved += len(values['solved_planning_time'])
        count_total += values['planning_tries']
        #sum_plan_length += sum( [plan_length for plan_length in values['plan_length']] )
        plan_lengths[problem_id] = values['plan_length']
        # sum_planning_time += sum( [planning_time for planning_time in values['solved_or_not_planning_time']] )
        sum_succ_planning_time += sum( [planning_time for planning_time in values['solved_planning_time']] )

        if 'plan_length' in samples_map[problem_id] and len(samples_map[problem_id]['plan_length']) > 0:
            problem_min_plan_length_map[problem_id] = min(samples_map[problem_id]['plan_length'])

#    return (sum_planning_time, sum_plan_length, count_solved, count_total, problem_min_plan_length_map)
    return (sum_succ_planning_time, plan_lengths, count_solved, count_total, problem_min_plan_length_map)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
