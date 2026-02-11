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


def listToStr(l: list) -> str:
    out = ''
    for item in l:
        if out:
            out += ', '
        out += str(item)
    return out


def getMethodRows(method) -> list[list[str]]:
    method_name = method['Name']
    out = []
    if 'TaskNetwork' in method and 'Tasks' in method['TaskNetwork'] and\
            len((subtasks := method['TaskNetwork']['Tasks'])) > 0:
        # print(f'method: "{method_name}", non-empty')
        
        method_str = f'\\multirow{{{len(subtasks)}}}{{2cm}}{{{method_name}}}'
        for idx, subtask in enumerate(subtasks):
            subtask_str = f'{subtask['Class']}({listToStr(subtask['Args'])})'
            if idx == 0:
                out.append( [None, method_str, subtask_str] )
            else:
                out.append( [None, None, subtask_str] )
    else:
        # print(f'method: "{method_name}", empty')
        method_str = f'{method_name}'
        subtask_str = '--'
        out.append( [None, method_str, subtask_str] )
    return out


def getTaskRows(task: dict[str, Any], methods: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    for method in methods:
        out += getMethodRows(method)
    return out

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="domain_visualization",
        description="Visualizes planning domain.",
    )
    parser.add_argument(
        "config",
        help="Config .json file path: absolute OR package://<pkg-name>/<relative-path>",
    )
    args = parser.parse_args(argv)

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

    config = _load_json_file(config_abs_path)

    planning_domain_path = _resolve_path_uri(config['planning_domain'], relative_path_root=config_dir_path)

    print('Loaded configuration:')
    print('  planning domain file:')
    print(f'    {planning_domain_path}')

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

    pd = PlanningDomain(planning_domain)

    txt = ''
    for task_class in pd.getComplexTasksClasses():
        task_def = pd.getComplexTaskDef(task_class)
        methods = pd.getMethodsForTaskClass(task_class)

        task_args_names = []
        for arg in task_def['Args']:
            task_args_names.append( arg['VarName'] )

        # Check if arg names match
        for method in methods:
            method_name = method['Name']
            m_task = method['Task']
            for subtask_arg_name, task_arg_name in zip(m_task['Args'], task_args_names):
                if subtask_arg_name != task_arg_name:
                    raise Exception(f'In method "{method_name}" names of task args '+\
                                    f'do not match: {m_task['Args']}, {task_args_names}')

        task_args_str = listToStr(task_args_names)
        # txt += f'\\multirow{{<<{task_class}_rows>>}}{{2cm}}'+\
        #         f'{{{task_class}({task_args_str})}} & <<method_name>> & <<first_subtask>> \\ \\cline{{2-3}}'

        task_rows = getTaskRows(task_def, methods)
        task_rows[0][0] = f'\\multirow{{{len(task_rows)}}}{{3cm}}'+\
                f'{{{task_class}({task_args_str})}}'
        # task_rows[-1][2] += '\\ \\cline{2-3}'

        for row in task_rows:
            cell0 = '' if row[0] is None else row[0]
            cell1 = '' if row[1] is None else row[1]
            txt += f'{cell0} & {cell1} & {row[2]} \\\\\n'
        txt += ' \\cline{2-3}\n'

        # txt += f'% task "{task_class}" methods: {len(methods)}\n'
        # task_rows = 0
        # for method in pd.getMethodsForTaskClass(task_class):
        #     method_name = method['Name']
        #     m_task = method['Task']
        #     m_task['Class']
        #     for subtask_arg_name, task_arg_name in zip(m_task['Args'], task_args_names):
        #         if subtask_arg_name != task_arg_name:
        #             raise Exception(f'In method "{method_name}" names of task args '+\
        #                             f'do not match: {m_task['Args']}, {task_args_names}')
        #     getMethodRows(method)

        # txt = txt.replace(f'<<{task_class}_rows>>', str(task_rows))
    txt = txt.replace('_', '\\_')
    print(txt)
    # 	\multirow{5}{2cm}{task1} & 1 & -- \\ \cline{2-3}
	# & \multirow{3}{2cm}{2} & subtask1 \\
	# & & subtask2 \\
	# & & subtask3 \\ \cline{2-3}
	# & 3 & subtask2 \\ \hline
	# task2 & & \\
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
