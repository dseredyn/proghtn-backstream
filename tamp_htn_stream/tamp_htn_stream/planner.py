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

import sys
import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict
from datetime import date, datetime
import os
from os import path
import shutil
import psutil

from ament_index_python.packages import get_package_share_directory

from .core import configure_planner, run_planner, clear_stats, PlanningDomain, ProblemTree

from .ebnf import GrammarParser, EBNFInterpreter
from .hsddl import VisitorHSDDL
from .visualization import PlanningVis

def _load_planning_domain(grammar_text, input_text):
    rules = GrammarParser(grammar_text).parse()

    # The first rule is the start rule
    start = next(iter(rules.keys()))

    # Remove comments
    input_text_no_com = ''
    for line in input_text.split('\n'):
        idx = line.find('#')
        if idx >= 0:
            input_text_no_com += line[0:idx] + '\n'
        else:
            input_text_no_com += line + '\n'

    no_skip_ws = True
    parser = EBNFInterpreter(rules, start, input_text_no_com, skip_ws=not no_skip_ws)

    try:
        tree = parser.parse()
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    out_obj = {
        "start": start,
        "tree": tree,
    }

    # print(tree.keys())
    visitor = VisitorHSDDL()
    planning_domain = visitor.visit(tree, 0)

    # Print planning domain as formatted json
    # json_string = json.dumps(planning_domain, indent=2)
    # print(json_string)

    # dump_kwargs = {"ensure_ascii": False}
    # pretty = True
    # if pretty:
    #     dump_kwargs.update({"indent": 2})
    # js = json.dumps(out_obj, **dump_kwargs)

    # if args.out:
    #     with open(args.out, "w", encoding="utf-8") as f:
    #         f.write(js)
    #         f.write("\n")
    # else:

    # print(js)

    return planning_domain


def _resolve_path_uri(uri, relative_path_root=None) -> str:
    if uri.startswith('package://'):
        # Package-relative path
        package_and_path = uri[10:]
        idx = package_and_path.find('/')
        package_name = package_and_path[0:idx]
        relative_path = package_and_path[idx+1:]
        share_dir = Path(get_package_share_directory(package_name))
        return str(share_dir / relative_path)
    elif uri.startswith('/'):
        # Absolute path
        return uri
    else:
        # Relative path
        if relative_path_root is None:
            raise Exception(f'Cannot process URI: {uri}; relative path is None.')
        if not relative_path_root.startswith('/'):
            raise Exception(f'Cannot process URI: {uri}; relative path is not absolutes.')
        return f'{relative_path_root}/{uri}' 
    
def _get_file_path(file_path: str) -> str:
    p = Path(file_path)
    assert p.is_absolute()
    return str(p.parent)

def _load_grammar_text(grammar_file_path) -> str:
    with open(grammar_file_path, 'r') as f:
        text = f.read()
    return text

def _load_json_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if text.strip() else {}
    return data

def _import_plugin(module_name: str):
    return importlib.import_module(module_name)

def createLogDir(log_dir, path_suffix):
    now = datetime.now() # current date and time

    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%H-%M-%S")

    subdir1_path = '{}/{}'.format(log_dir, date_dir)
    subdir2_path = '{}/{}-{}'.format(subdir1_path, time_dir, path_suffix)
    if not path.exists(subdir1_path):
        os.mkdir(subdir1_path)
    if not path.exists(subdir2_path):
        os.mkdir(subdir2_path)
    return subdir2_path

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="planner",
        description="Run planner with a dynamically loaded plugin and config from tamp_htn_stream_examples.",
    )
    parser.add_argument(
        "config",
        help="Config .json file path: absolute OR package://<pkg-name>/<relative-path>",
    )
    parser.add_argument(
        "--offline-test",
        help="Test suite .json file path: absolute OR package://<pkg-name>/<relative-path>",
    )
    args = parser.parse_args(argv)

    # TODO: read this from cmdline:
    out_vis_path_base = '/home/dseredyn/ws_tamp/visualization/planning'

    grammar_uri = 'package://tamp_htn_stream/data/grammar_hsddl.txt'
    grammar_file_path = _resolve_path_uri(grammar_uri)

    print('Using grammar:')
    print(f'  {grammar_uri}')
    print(f'  abs: {grammar_file_path}')

    config_abs_path = _resolve_path_uri(args.config)
    config_dir_path = _get_file_path(config_abs_path)

    print('Reading configuration:')
    print(f'  {args.config}')
    print(f'  abs: {config_abs_path}')

    print(f'Offline tests: {args.offline_test}')
    if args.offline_test is None:
        # TODO: add support for online planning: use ROS service as planner interface
        raise Exception('Online planning is not supported')
    # else:

    offline_test_abs_path = _resolve_path_uri(args.offline_test)
    offline_test_dir_path = _get_file_path(offline_test_abs_path)

    config = _load_json_file(config_abs_path)
    offline_test = _load_json_file(offline_test_abs_path)

    planning_domain_path = _resolve_path_uri(config['planning_domain'], relative_path_root=config_dir_path)

    print('Loaded configuration:')
    print('  planning domain file:')
    print(f'    {planning_domain_path}')

    print('  world model module:')
    print(f'    {config['world_model']}')

    print('Reading grammar')
    grammar_text = _load_grammar_text(grammar_file_path)

    print('Reading and parsing planning domain')

    p = Path(planning_domain_path)
    if not p.exists():
        raise Exception('File doe not exist')
    input_text = p.read_text(encoding="utf-8")

    planning_domain = _load_planning_domain(grammar_text, input_text)
    with open('/tmp/planning_domain.json', 'w') as f:
        f.write(json.dumps(planning_domain, indent=2, ensure_ascii=False))

    print(f'Importing world model module ({config['world_model']})')
    world_model_module = _import_plugin(config['world_model'])

    # Running planner
    configure_planner(world_model_module, config['world_model_data'], config_dir_path)

    pd = PlanningDomain(planning_domain)

    # Global planner parameters
    # max_iterations = offline_test['planner_parameters']['max_iterations']
    # visualization_verbosity_level = offline_test['planner_parameters']['visualization_verbosity_level']
    # repeats = offline_test['planner_parameters']['repeats']
    # use_backtracking = offline_test['planner_parameters']['use_backtracking']
    # generator_sampling_type = offline_test['planner_parameters']['generator_sampling_type']
    # test_run_id = offline_test['planner_parameters']['test_run_id']

    # Prepare test run
    tests_list = []
    for idx in range(offline_test['test_suite_repeats']):
        for test in offline_test['test_suite']:
            tests_list.append(test)

    max_RAM_usage_GB = int(offline_test['max_RAM_usage_GB'])

    for test in tests_list:

        if 'max_planning_time' in test:
            max_planning_time = test['max_planning_time']
        else:
            max_planning_time = None

        if 'max_iterations' in test:
            max_iterations = test['max_iterations']
        else:
            max_iterations = None

        visualization_verbosity_level = test['visualization_verbosity_level']
        use_backtracking = test['use_backtracking']
        generator_sampling_type = test['generator_sampling_type']
        test_run_id = test['test_run_id']

        for pp in offline_test['planning_problems']:
            if not pp['id'] in offline_test['active_tests']:
                continue
            # else:
            print(f'Running offline test for planning problem id: {pp['id']}')

            vm = psutil.virtual_memory()
            vm_total_GB = vm.total / (1024**3)
            vm_used_GB = vm.used / (1024**3)
            print('RAM usage:')            
            print(f'  Total:     {vm_total_GB:.2f} GiB')
            print(f'  Available: {vm.available / (1024**3):.2f} GiB')
            print(f'  Used:      {vm_used_GB:.2f} GiB')
            print(f'  Percent:   {vm.percent:.1f}%')

            if vm_used_GB > max_RAM_usage_GB:
                print('Exceeded maximum memory usage. Stopping.')
                return 0
            # else:

            # if 'max_iterations' in pp:
            #     max_iterations = pp['max_iterations']
            # print(f'max_iterations: {max_iterations}')

            # if 'visualization_verbosity_level' in pp:
            #     visualization_verbosity_level = pp['visualization_verbosity_level']

            print(f'  initial state: {pp['initial_state']}')
            initial_state_abs_path = _resolve_path_uri(pp['initial_state'], offline_test_dir_path)
            initial_state_json = _load_json_file(initial_state_abs_path)
            initial_state = world_model_module.parse_world_state(initial_state_json)

            print(f'  initial task network: {pp['initial_task_network']}')
            initial_task_network_abs_path = _resolve_path_uri(pp['initial_task_network'], offline_test_dir_path)
            initial_task_network_json = _load_json_file(initial_task_network_abs_path)
            initial_task_network = world_model_module.parse_task_network(initial_task_network_json)

            # Ablation tests: type of sampling
            world_model_module.set_generator_sampling_type( generator_sampling_type )

            pt = ProblemTree(initial_state, initial_task_network)
            plans = []
            stats = None
            exception = True
            try:
                clear_stats()
                plans, stats = run_planner(pd, world_model_module, pt,
                                           max_iterations, max_planning_time, use_backtracking)
                if not max_iterations is None:
                    stats.setValue('algorithm_run.max_iterations', max_iterations)
                if not max_planning_time is None:
                    stats.setValue('algorithm_run.max_planning_time', max_planning_time)
                stats.setValueObj('algorithm_run.problem_id', pp['id'])
                stats.setValueObj('algorithm_run.generator_sampling_type', generator_sampling_type)
                stats.setValueObj('algorithm_run.use_backtracking', use_backtracking)
                #stats.setValueObj('algorithm_run.repeat_idx', repeat_idx)
                stats.setValueObj('algorithm_run.test_run_id', test_run_id)
                stats.setValueObj('algorithm_input.problem', pp)
                exception = False
            # except Exception as e:
            #     print(e)
            #     exception = True
            finally:
                out_vis_path = createLogDir(out_vis_path_base, f'{pp['id']}_{test_run_id}')
                print(f'Saving planning results to {out_vis_path}')
                print(f'visualization_verbosity_level: {visualization_verbosity_level}')
                pv = PlanningVis(pd, world_model_module, pt, plans, stats, visualization_verbosity_level)
                pv.write(out_vis_path)
                shutil.copytree(out_vis_path, f'{out_vis_path_base}/latest', dirs_exist_ok=True)

            if exception:
                return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
