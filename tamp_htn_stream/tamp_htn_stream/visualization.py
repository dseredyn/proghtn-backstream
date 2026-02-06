#!/usr/bin/env python

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


import time
import os
import shutil
import json

from .core import PlanningDomain, ProblemTree, StatsMap

# from htn_geom_planner.svg_parser import SvgDocument
# from htn_geom_planner.planning_domain_lang_grammar4 import getDomainStrChunks
# from rcprg_system_common.common_classes import IndentInfo

def saveTreeInfo( path, pl, planning_time, planning_time_streams, test_case_id ):
    goal_nodes = pl.getGoalNodes()
    txt = 'node_selection_algorithm: {}\n'.format(pl.getNodeSelectionAlgorithm())
    txt += 'planning_id: {}\n'.format(pl.getPlanningId())
    txt += 'planning_time: {}\n'.format(planning_time)
    planning_time_streams_total = 0.0
    for stream_name in planning_time_streams:
        planning_time_streams_total += planning_time_streams[stream_name]
    txt += 'planning_time_streams: {}\n'.format(planning_time_streams_total)
    for stream_name in planning_time_streams:
        txt += 'planning_time_stream_{}: {}\n'.format(stream_name, planning_time_streams[stream_name])

    txt += 'test_case_id: {}\n'.format(test_case_id)
    txt += 'states: {}\n'.format(pl.getStatesCount())
    txt += 'nodes: {}\n'.format(pl.getNodesCount())
    txt += 'leafs: {}\n'.format(pl.getLeafsCount())
    txt += 'branch_nodes: {}\n'.format(pl.getBranchingNodesCount())
    txt += 'goal_nodes: {}\n'.format( len(goal_nodes) )
    for node_id in goal_nodes:
        path_to_root = pl.getPathToRoot(node_id)
        txt += 'goal_depth: {} {}\n'.format(node_id, len(path_to_root))
        plan = pl.getPlan(node_id)
        print(str(plan))

        actions_count = plan.getLength()
        # for anc_node_id in path_to_root:
        #     if pl.isActionNode(anc_node_id):
        #         actions_count += 1
        txt += 'actions_count: {} {}\n'.format(node_id, actions_count)

        filename = '{}/plan_{}.xml'.format(path, node_id)
        try:
            plan_xml = plan.serializeStr(use_indents=True)
            with open(filename, 'w') as f:
                f.write(plan_xml)
        except Exception as e:
            print('Could not save plan to file: {}'.format(filename))
            print(e)

        filename = '{}/plan_{}.tex'.format(path, test_case_id)
        try:
            plan_latex = plan.toLatex(test_case_id)
            with open(filename, 'w') as f:
                f.write(plan_latex)
        except Exception as e:
            print('Could not save plan to file: {}'.format(filename))
            print(e)

    print('saveTreeInfo:')
    print(txt)
    filename = '{}/tree_info.txt'.format(path)
    try:
        with open(filename, 'w') as f:
            f.write(txt)
    except Exception as e:
        print('Could not save to file: {}'.format(filename))
        print(e)

    filename2 = '{}/tree_info_{}_{}.txt'.format(path, pl.getNodeSelectionAlgorithm(), test_case_id)
    try:
        with open(filename2, 'w') as f:
            f.write(txt)
    except Exception as e:
        print('Could not save to file: {}'.format(filename2))
        print(e)

def parseLines(lines):
    # node_selection_algorithm: dfs_rnd_pr
    # planning_id: 1677683539_726469039
    # planning_time: 10.821586132
    # test_case_id: c
    # states: 9
    # nodes: 51
    # leafs: 1
    # branch_nodes: 0
    # goal_nodes: 1
    # goal_depth: 50 51
    # actions_count: 50 28
    result = {}
    for line in lines:
        l = line.strip()
        #print l
        if l.startswith('node_selection_algorithm:'):
            result['node_selection_algorithm'] = l[25:].strip()
        elif l.startswith('test_case_id:'):
            result['test_case_id'] = l[13:].strip()
        elif l.startswith('planning_time'):
            idx = l.index(':')
            item_name = l[0:idx]
            result[item_name] = float(l[idx+1:])
            #result['planning_time'] = float(l[14:])
        #elif l.startswith('planning_time_streams:'):
        #    result['planning_time_streams'] = float(l[22:])
        elif l.startswith('states:'):
            result['states'] = int(l[7:])
        elif l.startswith('nodes:'):
            result['nodes'] = int(l[6:])
        elif l.startswith('leafs:'):
            result['leafs'] = int(l[6:])
        elif l.startswith('branch_nodes:'):
            result['branch_nodes'] = int(l[13:])
        elif l.startswith('goal_nodes:'):
            result['goal_nodes'] = int(l[11:])
        elif l.startswith('goal_depth:'):
            items = l[11:].strip().split(' ')
            result['goal_depth'] = int(items[1])
        elif l.startswith('actions_count:'):
            items = l[14:].strip().split(' ')
            result['actions_count'] = int(items[1])
            result['goal_node_id'] = int(items[0])

    return result

def loadTreeInfo(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return parseLines(lines)

def getJsCodeInitROS():
    return '''
// Connecting to ROS
// -----------------
var ros = new ROSLIB.Ros({
    url : 'ws://localhost:9090'
});
ros.on('connection', function() {
    console.log('Connected to websocket server.');
    msg_p = document.getElementById("message");
    if (msg_p != null) {
        msg_p.innerHTML = "Connected to websocket server.";
    }
});
ros.on('error', function(error) {
    console.log('Error connecting to websocket server: ', error);
    msg_p = document.getElementById("message");
    if (msg_p != null) {
        msg_p.innerHTML = "Error connecting to websocket server. Please run:<br><i>roslaunch rosbridge_server rosbridge_websocket.launch</i><br>and refresh this page";
    }
});
ros.on('close', function() {
    console.log('Connection to websocket server closed.');
});'''

def nameInList(name, name_list):
    if name in name_list:
        return True
    for nn in name_list:
        if nn.endswith('*'):
            prefix = nn[:-1]
            if name.startswith(prefix):
                return True
    return False

# def readNodesLocations(dot_filename, node_id_int):
#     result = []
#     known_node_types = ['node', 'tp', 'task', 'state']
#     if dot_filename.endswith('.plain'):
#         with open(dot_filename, 'rt') as f:
#             lines = f.readlines()
#             fields = lines[0].split()
#             assert fields[0] == 'graph'
#             graph_size = (float(fields[2]), float(fields[3]))
#             for line in lines:
#                 fields = line.split()
#                 if fields[0] == 'node':
#                     if fields[1][5:].endswith('_fork'):
#                         node_idx = int(fields[1][5:-5])
#                     elif fields[1][5:].endswith('_data'):
#                         continue
#                     else:
#                         #node_idx = int(fields[1][5:])
#                         node_idx = int(fields[1])
#                     px = float(fields[2])
#                     py = float(fields[3])
#                     sx = float(fields[4])
#                     sy = float(fields[5])
#                     result.append( (node_idx, px, py, sx, sy) )
#     elif dot_filename.endswith('.svg'):
#         svg = SvgDocument(dot_filename, 1)
#         graph_size = (svg.width, svg.height)
#         tx = None
#         ty = None
#         for object_id, obj in svg.object_id_dict.items():
#             if obj.getType() == 'g':
#                 assert obj.class_name != None
#                 if obj.class_name == 'graph':
#                     tx = obj.transform.translate_x
#                     ty = obj.transform.translate_y
#                 elif obj.class_name == 'cluster':
#                     title_objs = obj.getAllChildren(['title'])
#                     assert len(title_objs) == 1
#                     assert title_objs[0].getType() == 'title'
#                     title = title_objs[0].name_str
#                     assert title.startswith('cluster_')
#                     node_idx = int(title[8:])

#                     polygon_objs = obj.getAllChildren(['polygon'])
#                     assert len(polygon_objs) == 1
#                     assert polygon_objs[0].getType() == 'polygon'
#                     min_x = float('inf')
#                     max_x = float('-inf')
#                     min_y = float('inf')
#                     max_y = float('-inf')
#                     for pt in polygon_objs[0].points:
#                         min_x = min(min_x, pt[0])
#                         max_x = max(max_x, pt[0])
#                         min_y = min(min_y, pt[1])
#                         max_y = max(max_y, pt[1])
#                     px = (min_x+max_x)/2
#                     py = (min_y+max_y)/2
#                     sx = max_x - min_x
#                     sy = max_y - min_y
#                     result.append( (node_idx, px, py, sx, sy) )
#                 elif obj.class_name == 'node':
#                     title_objs = obj.getAllChildren(['title'])
#                     assert len(title_objs) == 1
#                     assert title_objs[0].getType() == 'title'
#                     title = title_objs[0].name_str
#                     node_type = None
#                     for node_type_str in known_node_types:
#                         if title.startswith(node_type_str+'_'):
#                             node_type = node_type_str
#                             #node_idx = int(title[len(node_type_str)+1:])
#                             if node_id_int:
#                                 node_idx = int(title[len(node_type_str)+1:])
#                             else:
#                                 node_idx = title[len(node_type_str)+1:]

#                     #if title.startswith('node_'):
#                     #    node_type = 'node'
#                     #    node_idx = int(title[5:])
#                     #elif title.startswith('tp_'):
#                     #    node_type = 'tp'
#                     #    node_idx = int(title[3:])
#                     #elif title.startswith('task_'):
#                     #    node_type = 'task'
#                     #    node_idx = int(title[5:])
#                     #elif title.startswith('state_'):
#                     #    node_type = 'state'
#                     #    node_idx = int(title[5:])
#                     #else:
#                     if node_type is None:
#                         raise Exception('Unknown type of node: "' + title + '"')
#                     polygon_objs = obj.getAllChildren(['polygon'])
#                     ellipse_objs = obj.getAllChildren(['ellipse'])
#                     if len(polygon_objs) > 0:
#                         assert polygon_objs[0].getType() == 'polygon'
#                         min_x = float('inf')
#                         min_y = float('inf')
#                         max_x = float('-inf')
#                         max_y = float('-inf')
#                         for poly in polygon_objs:
#                             for pt in poly.points:
#                                 min_x = min(min_x, pt[0])
#                                 min_y = min(min_y, pt[1])
#                                 max_x = max(max_x, pt[0])
#                                 max_y = max(max_y, pt[1])
#                     elif len(ellipse_objs) > 0:
#                         min_x = ellipse_objs[0].cx-ellipse_objs[0].rx
#                         max_x = ellipse_objs[0].cx+ellipse_objs[0].rx
#                         min_y = ellipse_objs[0].cy-ellipse_objs[0].ry
#                         max_y = ellipse_objs[0].cy+ellipse_objs[0].ry
#                     else:
#                         raise Exception('Unknown shape od node')
#                     px = (min_x+max_x)/2
#                     py = (min_y+max_y)/2
#                     sx = max_x - min_x
#                     sy = max_y - min_y
#                     result.append( (node_type, node_idx, px, py, sx, sy) )
#                 elif obj.class_name == 'edge':
#                     # Get rectangle frame of edge title, if it is present
#                     polygon_objs = obj.getAllChildren(['polygon'])
#                     if len(polygon_objs) > 0:
#                         assert polygon_objs[0].getType() == 'polygon'
#                         title_objs = obj.getAllChildren(['title'])
#                         assert len(title_objs) == 1
#                         assert title_objs[0].getType() == 'title'
#                         title = title_objs[0].name_str
#                         #print title
#                         sep_idx = title.find('->')
#                         if sep_idx < 0:
#                             continue
#                         #print title
#                         node_src_str = title[0:sep_idx]
#                         node_dst_str = title[sep_idx+2:]
#                         if node_src_str.startswith('node_') and node_dst_str.startswith('node_'):
#                             src_node_id = int(node_src_str[5:])
#                             dst_node_id = int(node_dst_str[5:])

#                             #print 'edge', src_node_id, dst_node_id

#                             min_x = float('inf')
#                             min_y = float('inf')
#                             max_x = float('-inf')
#                             max_y = float('-inf')
#                             for poly in polygon_objs:
#                                 if poly.fill_str != 'none':
#                                     continue
#                                 for pt in poly.points:
#                                     min_x = min(min_x, pt[0])
#                                     min_y = min(min_y, pt[1])
#                                     max_x = max(max_x, pt[0])
#                                     max_y = max(max_y, pt[1])
#                             px = (min_x+max_x)/2
#                             py = (min_y+max_y)/2
#                             sx = max_x - min_x
#                             sy = max_y - min_y
#                             node_type = 'edge_title'
#                             result.append( (node_type, (src_node_id, dst_node_id), px, py, sx, sy) )
#         # Rescale positions of all nodes
#         for idx in range(len(result)):
#             node_type, node_idx, px, py, sx, sy = result[idx]
#             result[idx] = (node_type, node_idx, px+tx, graph_size[1]-(py+ty), sx, sy)
#     else:
#         raise Exception('Unsupported dot graph extension for file: "' + dot_filename + '"')

#     return graph_size, result

# dict_nodes_dir contains dictionary of node_id -> href for node_frame html frame (left)
# dict_nodes_vis contains dictionary of node_id -> href for vis_frame html frame (right, top)
def generateScriptForHtnGraph(dict_nodes_dir, dict_nodes_vis, graph_size, nodes_locations, img_size):
    # Clusters may overlap, so their order in if-else is important.
    # Sort clusters wrt. their area.
    nodes_locations_sorted = sorted(nodes_locations, key=lambda x: x[3]*x[4])

    # There is 3px margin from dot and 1px margin for html image border
    margin = 4
    if graph_size[0] == 0 or graph_size[1] == 0:
        scale_x = 1.0
        scale_y = 1.0
    else:
        scale_x = (img_size[0]-2*margin)/graph_size[0]
        scale_y = (img_size[1]-2*margin)/graph_size[1]

    list_xmin = []
    list_xmax = []
    list_ymin = []
    list_ymax = []
    list_nodes_dir = []
    list_nodes_vis = []
    for loc in nodes_locations_sorted:
        node_type, node_idx, px, py, sx, sy = loc

        xmin = (px - sx/2) * scale_x + margin
        xmax = max( (px + sx/2) * scale_x, xmin+4 ) + margin
        ymin = img_size[1] - (py + sy/2) * scale_y - margin
        ymax = max( img_size[1] - (py - sy/2) * scale_y, ymin+4 ) - margin
        list_xmin.append(xmin)
        list_xmax.append(xmax)
        list_ymin.append(ymin)
        list_ymax.append(ymax)

        if node_idx in dict_nodes_dir[node_type]:
            list_nodes_dir.append( dict_nodes_dir[node_type][node_idx] )
        else:
            list_nodes_dir.append( None )

        if node_idx in dict_nodes_vis[node_type]:
            list_nodes_vis.append( dict_nodes_vis[node_type][node_idx] )
        else:
            list_nodes_vis.append( None )
    str_list_nodes_dir = ''
    for item in list_nodes_dir:
        if item is None:
            val = 'null'
        else:
            val = '\'' + str(item) + '\''
        if str_list_nodes_dir != '':
            str_list_nodes_dir += ', '
        str_list_nodes_dir += val

    str_list_nodes_vis = ''
    for item in list_nodes_vis:
        if item is None:
            val = 'null'
        else:
            val = '\'' + str(item) + '\''
        if str_list_nodes_vis != '':
            str_list_nodes_vis += ', '
        str_list_nodes_vis += val

    html = '<script>\n'
    html += 'function printMousePos(event) {\n'
    html += '    var list_x_min = ' + str(list_xmin) + ';\n'
    html += '    var list_x_max = ' + str(list_xmax) + ';\n'
    html += '    var list_y_min = ' + str(list_ymin) + ';\n'
    html += '    var list_y_max = ' + str(list_ymax) + ';\n'
    html += '    var list_nodes_dir = [' + str_list_nodes_dir + '];\n'
    html += '    var list_nodes_vis = [' + str_list_nodes_vis + '];\n'
    html += '    var c = document.getElementById("myCanvas");\n'
    html += '    var ctx = c.getContext("2d");\n'
    html += '    ctx.fillStyle = \'rgba(1,0,0,0.4)\';\n'
    html += '    try {\n'
    html += '        for (i = 0; i < ' + str(len(list_xmin)) + '; i++) {\n'
    html += '            if (event.offsetX > list_x_min[i] && event.offsetX < list_x_max[i] && event.offsetY > list_y_min[i] && event.offsetY < list_y_max[i]) {\n'
    html += '                if (list_nodes_dir[i] != null) {\n'
    html += '                    top.frames[\'node_frame\'].location.href = list_nodes_dir[i];\n'
    html += '                }\n'
    html += '                if (list_nodes_vis[i] != null) {\n'
    html += '                    top.frames[\'vis_frame\'].location.href = list_nodes_vis[i];\n'
    html += '                }\n'
    html += '                ctx.clearRect(0,0,' + str(img_size[0]) + ',' + str(img_size[1]) + ')\n'
    html += '                ctx.fillRect(list_x_min[i], list_y_min[i], list_x_max[i]-list_x_min[i], list_y_max[i]-list_y_min[i]);\n'
    html += '                ctx.stroke();\n'
    html += '                break;\n'
    html += '            }\n'
    html += '        }\n'
    html += '    }\n'
    html += '    catch(error) {\n'
    html += '      console.error(error);\n'
    html += '        var coor = "X coords: " + event.offsetX + ", Y coords: " + event.offsetY;\n'
    html += '        document.getElementById("demo").innerHTML = "When using Firefox change security.fileuri.strict_origin_policy to false (address: about:config)";\n'
    html += '      console.info("When using Firefox change security.fileuri.strict_origin_policy to false (address: about:config)")\n'
    html += '    }\n'

    html += '}\n'
    html += 'document.getElementById(\'htn_dot_img\').addEventListener("click", printMousePos);\n'
    html += 'document.getElementById(\'myCanvas\').addEventListener("click", printMousePos);\n'
    html += '</script>\n'

    return html


class PlanningVis:
    def __init__(self, pd, world_model_module, pt, plans: list, stats: None|StatsMap, verbosity_level: str):
        assert isinstance(pd, PlanningDomain)
        assert isinstance(pt, ProblemTree)
        assert verbosity_level in ['none', 'low', 'medium', 'high']
        self._pd = pd
        self._world_model_module = world_model_module
        self._pt = pt
        self._plans = plans
        self._stats = stats
        self._verbosity_level = verbosity_level

    def getPath(self, path_type, path_name, path_args):
        path_map = {}
        if path_name == 'search_node_html':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            path_map['index'] = 'search_nodes/{}/{}.html'.format(node_subdir, node_id)
        elif path_name == 'search_node_dot':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            path_map['index'] = 'search_nodes/{}/{}'.format(node_subdir, node_id)
        elif path_name == 'search_arrow_html':
            arrow_id = path_args[0]
            path_map['index'] = 'search_arrows/{}.html'.format(arrow_id)
        elif path_name == 'task_html':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            task_id = path_args[1]
                #.replace('+', 'lock_').replace('-', 'unlock_')
            path_map['index'] = 'search_nodes/{}/{}_task_{}.html'.format(node_subdir, node_id, task_id)
        elif path_name == 'js_task_vis':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            task_id = path_args[1]
                #.replace('+', 'lock_').replace('-', 'unlock_')
            path_map['index'] = 'search_nodes/{}/{}_task_{}.js'.format(node_subdir, node_id, task_id)
        elif path_name == 'main_html':
            path_map['index'] = 'main.html'
        elif path_name == 'data_html':
            path_map['index'] = 'data.html'
        elif path_name == 'blank_html':
            path_map['index'] = 'blank.html'
        elif path_name == 'search_dot':
            path_map['index'] = './search_graph'
        elif path_name == 'search_graph_html':
            path_map['index'] = './search_graph.html'
        elif path_name == 'cached_data_dir':
            state_id = path_args[0]
            key = path_args[1]
            path_map['index'] = 'cached_data/{}/{}'.format( state_id, key )
        elif path_name == 'cached_data_html':
            return self.getPath(path_type, 'cached_data_dir', path_args) + '/data.html'
        elif path_name == 'cached_data_js':
            return self.getPath(path_type, 'cached_data_dir', path_args) + '/script.js'
        elif path_name == 'global_cached_data_dir':
            key = path_args[0]
            path_map['index'] = 'global_cached_data/{}'.format( key )
        elif path_name == 'global_cached_data_html':
            return self.getPath(path_type, 'global_cached_data_dir', path_args) + '/data.html'
        elif path_name == 'states_graph_dot':
            path_map['index'] = './states_graph'
        elif path_name == 'search_graph_dot':
            path_map['index'] = './search_graph'
        elif path_name == 'search_graph_dot_small':
            path_map['index'] = './search_graph_small'
        elif path_name == 'state_node_html':
            state_id = path_args[0]
            path_map['index'] = './states/{}.html'.format(state_id)
        elif path_name == 'method_html':
            method_name = path_args[0]
            path_map['index'] = './methods/{}.html'.format(method_name)
        elif path_name == 'method_dot':
            method_name = path_args[0]
            path_map['index'] = './methods/{}'.format(method_name)
        elif path_name == 'domain_html':
            path_map['index'] = './domain.html'
        elif path_name == 'domain_dot':
            path_map['index'] = './domain.dot'
        # elif path_name == 'solutions_html':
        #     path_map['index'] = './solutions.html'
        elif path_name == 'solution_html':
            node_id = path_args[0]
            path_map['index'] = './solutions/{}.html'.format(node_id)
        elif path_name == 'solution_vis':
            node_id = path_args[0]
            path_map['index'] = './solutions/vis_{}.html'.format(node_id)
        elif path_name == 'solution_dot':
            node_id = path_args[0]
            path_map['index'] = './solutions/{}'.format(node_id)
        elif path_name == 'task_def_html':
            task_name = path_args[0]
            path_map['index'] = './tasks/{}'.format(task_name)
        elif path_name == 'js_init_ros':
            path_map['index'] = './init_ros.js'
        elif path_name == 'js_init_state_pub':
            path_map['index'] = './init_state_pub.js'
        elif path_name == 'js_init_task_pub':
            path_map['index'] = './init_task_pub.js'
        elif path_name == 'generator_html':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            path_map['index'] = 'search_nodes/{}/{}_generator.html'.format(node_subdir, node_id)
        elif path_name == 'js_init_generator_pub':
            path_map['index'] = './init_generator_pub.js'
        elif path_name == 'js_generator_vis':
            node_id = path_args[0]
            node_subdir = self.__getNodeSubdir(node_id)
            path_map['index'] = 'search_nodes/{}/{}_generator.js'.format(node_subdir, node_id)
        elif path_name == 'metadata_xml':
            path_map['index'] = './metadata.xml'
        elif path_name == 'stats_json':
            path_map['index'] = './stats.json'
        elif path_name == 'world_model':
            path_map['index'] = './world_model.xml'
        else:
            raise Exception('Unknown path_name: "' + str(path_name) + '"')

        path_map['abs'] = self.__output_path + '/' + path_map['index']
        path_map['search_node'] = '../../' + path_map['index']
        path_map['search_arrow'] = '../' + path_map['index']
        path_map['data'] = '../../' + path_map['index']
        path_map['states'] = '../' + path_map['index']
        path_map['methods'] = '../' + path_map['index']
        path_map['tasks'] = '../' + path_map['index']
        path_map['solutions_dir'] = '../' + path_map['index']
        path_map['cached_data_item_dir'] = '../../../' + path_map['index']

        if path_type in path_map:
            return path_map[path_type]
        else:
            raise Exception('Unknown path_type: "' + str(path_type) + '" for "' + str(path_name) + '"')

    def write(self, output_path):
        self.__output_path = output_path

        remove_all = False
        try:
            if remove_all:
                shutil.rmtree(self.__output_path + '/html')
        except OSError as e:
            pass
        os.makedirs(self.__output_path + '/search_nodes')
        os.makedirs(self.__output_path + '/states')
        os.makedirs(self.__output_path + '/cached_data')
        os.makedirs(self.__output_path + '/global_cached_data')
        os.makedirs(self.__output_path + '/methods')
        os.makedirs(self.__output_path + '/solutions')

        self.__writeStats()
        self.__writeSolutions()
        self.__writeCommonScripts()

        # t1 = time.time()
        self.__writeFramesPage()
        # t2 = time.time()
        self.__writeMainPage()
        # t3 = time.time()

        if self._verbosity_level == 'none':
            return
        # else:

        write_search_nodes = True
        if write_search_nodes:
            self.__writeSearchNodes()
        else:
            print('WARNING: writing of search nodes is disabled!')

        # t5 = time.time()
        # t6 = time.time()
        #self.__writeSearchGraph()
        # t7 = time.time()

    def __writeStats(self):
        if not self._stats is None:
            with open(self.getPath('abs', 'stats_json', None), 'w') as f:
                f.write(json.dumps(self._stats.toJson(), indent=2, ensure_ascii=False))

    def __writeCommonScripts(self):
        with open(self.getPath('abs', 'js_init_ros', None), 'w') as f:
            f.write( getJsCodeInitROS() )

    def __writePlans(self):
        pass

    def __writeMetadata(self):
        pass
        # if not self.__wm_metadata is None:
        #     with open(self.getPath('abs', 'metadata_xml', None), 'w') as f:
        #         f.write( self.__wm_metadata.serializeStr() )

    def __writeSolutions(self):
        print('PlanningVis: writing solutions')
        canvas_cover_style = ''
        for idx, plan in enumerate(self._plans):
            # html += f'<p><a href="{solutions_path}" target="node_frame" >Solutions ({len(self._plans)})</a></p>\n'

            html = '<!DOCTYPE html>\n'
            html += '<html>\n'
            html += '<style>\n'
            html += '  table, th, td {\n'
            html += '    border: 1px solid black;\n'
            html += '    border-collapse: collapse;\n'
            html += '  }\n'
            html += canvas_cover_style
            html += '  td.pad10 {\n'
            html += '    padding: 10px;\n'
            html += '  }\n'
            html += '  td.hvr {\n'
            html += '    padding: 10px;\n'
            html += '    background-color: #ffff80;\n'
            html += '  }\n'
            html += '  td.hvr:hover {\n'
            html += '    background-color: #00ff00;\n'
            html += '    color: #000000;\n'
            html += '  }\n'
            html += '</style>\n'
            html += '<body>\n'
            html += f'<p>Solution #{idx}</p>'

            html += '<TABLE><TR><TD><b>class</b></TD><TD><b>parameters</b></TD></TR>\n'
            for task, param_list in plan:
                arg_values: list[str] = []
                colors: list[str] = []
                for param_data in param_list:
                    value_str = self._world_model_module.format_as_text(param_data)
                    arg_values.append(value_str)
                    colors.append( 'black' )
                html += f'<TR><TD>{task['Class']}</TD>'+\
                            f'<TD>{listToStrTooltip(task['Args'], arg_values, colors)}</TD></TR>\n'
            html += '</TABLE></p>\n'


            # TODO: add visualization of motion for whole plan

                # html += print(f'  ({task['Class']} {task['Args']})')
            # html += '<p><img src="{}.svg" /></p>\n'.format( self.getPath('solutions_dir', 'solution_dot', (node_id,)) )
            html += '</body>\n'
            html += '</html>\n'

            with open(self.getPath('abs', 'solution_html', (idx,)), 'w') as f:
                f.write(html)

        return
        # # TODO
        # # Write individual page for each solution
        # goal_nodes = self.__pg.getGoalNodes()
        # for node_id in goal_nodes:
        #     plan = self.__pg.getPlan(node_id)
        #     dot = plan.getDot(show_all_values=True)

        #     self.__dot_interface.write(dot, self.getPath('abs', 'solution_dot', (node_id,)))

        #     html = '<!DOCTYPE html>\n'
        #     html += '<html>\n'
        #     html += '<body>\n'
        #     html += '<p>Solution for node {}</p>'.format(node_id)
        #     html += '<p><img src="{}.svg" /></p>\n'.format( self.getPath('solutions_dir', 'solution_dot', (node_id,)) )
        #     html += '</body>\n'
        #     html += '</html>\n'

        #     with open(self.getPath('abs', 'solution_html', (node_id,)), 'w') as f:
        #         f.write(html)

        # # Write html with a list of all methods
        # html = '<!DOCTYPE html>\n'
        # html += '<html>\n'
        # html += '<style>\n'
        # html += '  table, th, td {\n'
        # html += '    border: 1px solid black;\n'
        # html += '    border-collapse: collapse;\n'
        # html += '  }\n'
        # html += '  th, td {\n'
        # html += '    padding: 5px;\n'
        # html += '  }\n'
        # html += '</style>\n'
        # html += '<script>'
        # html += self.__getMainLinkScript('index')
        # html += '</script>\n'
        # html += '<body>\n'
        # html += self.__getMainLink('index')
        # html += '<h2>Solutions</h2>\n'
        # if len(goal_nodes) == 0:
        #     html += '<p>no solutions</p>'
        # else:
        #     for idx, node_id in enumerate(goal_nodes):
        #         node = self.__pg.getTree().getNode(node_id)
        #         html += '<p>{}. <a href="{}" target="vis_frame">solution at node {}, state {}</a></p>'.format(
        #             idx+1, self.getPath('index', 'solution_html', (node_id,)), node_id, node.getStateId())

        # html += '</body>\n'
        # html += '</html>\n'
        # with open( self.getPath('abs', 'solutions_html', None), 'w' ) as f:
        #     f.write(html)

    def __writeDataHtml(self, abs_html_path, value):
        html = '<!DOCTYPE html>\n'
        html += '<html>\n'
        html += '<body>\n'
        html += '<p>TODO</p>'
        html += '</body>\n'
        html += '</html>\n'

        # Write html page for this particular node
        with open( abs_html_path, 'w' ) as f:
            f.write(html)

    def __writeMethods(self):
        pass
        # html = '<!DOCTYPE html>\n'
        # html += '<html>\n'
        # html += '<style>\n'
        # html += 'table {\n'
        # html += '  border-collapse: collapse;\n'
        # html += '}\n'
        # html += 'td {\n'
        # html += '  padding-top: 0;\n'
        # html += '  padding-bottom: 0;\n'
        # html += '  padding-left: 10px;\n'
        # html += '  padding-right: 10px;\n'
        # html += '}\n'
        # html += 'pre {\n'
        # html += '  display: block;\n'
        # html += '  font-family: monospace;\n'
        # html += '  white-space: pre;\n'
        # html += '  margin: 0 0;\n'
        # html += '}\n'
        # html += '</style>\n'
        # html += '<body>'
        # html += '<p>TODO: better visualization</p>'

        # html += '</body>'
        # html += '</html>\n'

        # with open( self.getPath('abs', 'domain_html', None), 'w' ) as f:
        #     f.write(html)

        # # dot = self.__pd.getMethodsDot()
        # # self.__dot_interface.write(dot, self.getPath('abs', 'domain_dot', None))

        # # with open(self.getPath('abs', 'domain_dot', None), 'w') as f:
        # #     f.write(dot)

        # # TODO: write decision trees
        # return

    # def __getCanvasCoverStyle(self, width, height):
    #     html = '    .outsideWrapper{ \n'
    #     html += '        width:{}px; height:{}px; \n'.format(width, height)
    #     html += '        margin:0px 0px; \n'
    #     html += '        border:0px solid blue;}\n'
    #     html += '    .insideWrapper{ \n'
    #     html += '        width:100%; height:100%; \n'
    #     html += '        position:relative;}\n'
    #     html += '    .coveredImage{ \n'
    #     html += '        width:100%; height:100%; \n'
    #     html += '        position:absolute; top:0px; left:0px;\n'
    #     html += '    }\n'
    #     html += '    .coveringCanvas{ \n'
    #     html += '        width:100%; height:100%; \n'
    #     html += '        position:absolute; top:0px; left:0px;\n'
    #     html += '        background-color: rgba(0,0,0,0);\n'
    #     html += '    }\n'
    #     return html

    def __getNodeSubdir(self, node_id: int) -> str:
        return str(int(node_id/1000))

    def __writeSearchNodes(self):

        # Compute task groups map
        task_dec_groups_map = {}

        task_groups_map = {}

        # Write all task networks
        added_subfolders = set()
        #nodes_written = 0
        map_graph_size = {}
        map_canvas_cover_style = {}
        map_htn_graph_html_script = {}

        deepest_node_id = self._pt.getDeepestNodeId()
        deepest_node_path = self.getPath('search_node', 'search_node_html', (deepest_node_id,))
        deepest_node_state_vis_path = self.getPath('search_node', 'cached_data_html', (deepest_node_id, 'vis'))
        deepest_node_text = '{}'.format(deepest_node_id)
        deepest_onclick_script = f'top.frames[\'node_frame\'].location.href=\'{deepest_node_path}\';'+\
                        f'top.frames[\'vis_frame\'].location.href=\'{deepest_node_state_vis_path}\';'

        nodes_written = 0
        #added_subfolders = set()
        # nodes = self.__pg.getTree().getNodes()
        #for node_id in sorted(self.__pg._ProgressionSearch__nodes.keys()):
        # for node_id in sorted(nodes.keys()):
        for node_id, node in self._pt.getAllNodes().items():
            nodes_written += 1
            if nodes_written%200 == 0:
                print('nodes_written: {}'.format(nodes_written))

            node_subdir = self.__getNodeSubdir(node_id)
            if not node_subdir in added_subfolders:
                added_subfolders.add(node_subdir)
                os.makedirs('{}/search_nodes/{}'.format(self.__output_path, node_subdir))

            # h_node_id = self.__pg.nodeId(node_id)
            # task_id_list = node.getTaskNetwork().getTaskIds()
            # #print('__writeSearchNodes: node_id: {}, task_id_list: {}'.format(node_id, task_id_list))

            # if node.getTaskNetwork().isEmpty() or not generate_tn_dot:
            #     map_graph_size[node_id] = (None, None)
            #     map_htn_graph_html_script[node_id] = ''
            #     map_canvas_cover_style[node_id] = ''
            # else:
            #     abs_path_graph = self.getPath('abs', 'search_node_dot', (node_id,))
            #     dot = self.__pg.getNodeDot( h_node_id )
            #     #print('task network dot:')
            #     #print(dot)
            #     self.__dot_interface.write(dot, abs_path_graph)
            #     graph_size, nodes_locations = readNodesLocations(abs_path_graph+'.svg', False)
            #     #print('  nodes_locations: {}'.format(nodes_locations))
            #     map_graph_size[node_id] = graph_size
            #     graph_width, graph_height = graph_size

            #     dict_nodes_dir = {'task':{}}
            #     dict_nodes_vis = {'task':{}}
            #     for task_id in task_id_list:
            #         #task_id_simple = task_id[1].replace('+', 'lock_').replace('-', 'unlock_')
            #         dict_nodes_vis['task'][task_id[1]] = self.getPath('search_node', 'task_html', (node_id, task_id[1]))

            #     map_htn_graph_html_script[node_id] = generateScriptForHtnGraph(dict_nodes_dir, dict_nodes_vis, graph_size, nodes_locations, graph_size)
            #     map_canvas_cover_style[node_id] = self.__getCanvasCoverStyle(graph_width, graph_height)

            #     graph_width, graph_height = graph_size
            #     htn_graph_html_script = map_htn_graph_html_script[node_id]
            #     canvas_cover_style = map_canvas_cover_style[node_id]
            canvas_cover_style = ''
            # task_id_list = node.getTaskNetwork().getTaskIds()
            # h_node_id = self.__pg.nodeId(node_id)

            html = '<!DOCTYPE html>\n'
            html += '<html>\n'
            html += '<style>\n'
            html += '  table, th, td {\n'
            html += '    border: 1px solid black;\n'
            html += '    border-collapse: collapse;\n'
            html += '  }\n'
            html += canvas_cover_style
            html += '  td.pad10 {\n'
            html += '    padding: 10px;\n'
            html += '  }\n'
            html += '  td.hvr {\n'
            html += '    padding: 10px;\n'
            html += '    background-color: #ffff80;\n'
            html += '  }\n'
            html += '  td.hvr:hover {\n'
            html += '    background-color: #00ff00;\n'
            html += '    color: #000000;\n'
            html += '  }\n'
            html += '</style>\n'
            html += '<script type="text/javascript">\n'
            html += self.__getMainLinkScript('search_node')

            html += '</script>\n'
            html += '<body>\n'
            html += self.__getMainLink('search_node')
            parent_node_id = self._pt.getParentNodeId( node_id )
            if parent_node_id is None:
                # html += '<p><TABLE><TR><TD class="pad10"><B>Parent node : none</B></TD></p>'
                #\
                #        '<TD class="hvr">None</TD></TR></TABLE></p>\n'
                html += f'<p><TABLE><TR><TD class="pad10"'+\
                        f' >Parent node: none</TD>'+\
                        f'<TD class="hvr"'+\
                        f' onclick="{deepest_onclick_script}">Deepest node: {deepest_node_text}</TD>'+\
                            '</TR></TABLE></p>\n'
            else:
                node_text = '{}'.format(parent_node_id)

                parent_node_path = self.getPath('search_node', 'search_node_html', (parent_node_id,))
                parent_node_state_vis_path = self.getPath('search_node', 'cached_data_html', (parent_node_id, 'vis'))
                onclick_script = f'top.frames[\'node_frame\'].location.href=\'{parent_node_path}\';'+\
                                f'top.frames[\'vis_frame\'].location.href=\'{parent_node_state_vis_path}\';'
                # html += f'<p><TABLE><TR><TD class="pad10"><B>Parent node</B></TD><TD class="hvr"'+\
                #         f' onclick="{onclick_script}">{node_text}</TD></TR></TABLE></p>\n'
                html += f'<p><TABLE><TR><TD class="hvr"'+\
                        f' onclick="{onclick_script}">Parent node: {node_text}</TD>'+\
                        f'<TD class="hvr"'+\
                        f' onclick="{deepest_onclick_script}">Deepest node: {deepest_node_text}</TD>'+\
                            '</TR></TABLE></p>\n'

            # Mark nodes on hte path to solution
            if node.subtree_has_solution:
                node_text = f'{node_id} (S)'
            else:
                node_text = f'{node_id}'

            # Every node has its own state
            state_id = node_id

            if node.isClosed():
                open_closed_str = 'closed'
            else:
                open_closed_str = 'open'
            p10cl = 'class="pad10"'
            html += f'<p><table><tr><td {p10cl}><b>Current node</b></td>'+\
                    f'<td {p10cl}>{node_text}</td>'+\
                    f'<td {p10cl}>depth: {node.depth}</td>'+\
                    f'<td {p10cl}>sub nodes: {node.subtree_nodes_count}</td>'+\
                    f'<td {p10cl}>{open_closed_str}</td></tr></table></p>\n'

            backtrack_results = node.getBacktrackSearchResults()
            if len(backtrack_results) > 0:
                html += '<h3>Backtrack search results:</h3>'
                for res in backtrack_results:
                    html += f'{res}<br>\n'
                html += '<p>'


            html += '<h3>Child nodes:</h3>'
            html += '<p>'

            html += '<TABLE><TR><TH rowspan="2"><b>node</b></TH><TH colspan="3"><b>subtree</b></TD></TR>'
            html += '<TR><TH><b>max depth</b></TH><TH><b>nodes</b></TH><TH><b>open</b></TH></TR>'
            child_nodes = self._pt.getChildrenNodesIds(node_id)
            child_nodes = sorted(child_nodes, key=lambda x: self._pt.getNode(x).subtree_max_depth, reverse=True)
            for child_node_id in child_nodes:
                child_node = self._pt.getNode(child_node_id)
                changes_str = ''
                if child_node.subtree_has_solution:
                    node_html = f'{child_node_id} (S)'
                else:
                    node_html = f'{child_node_id}'
                child_node_path = self.getPath('search_node', 'search_node_html', (child_node_id,))
                child_node_state_vis_path = self.getPath('search_node', 'cached_data_html', (child_node_id, 'vis'))
                on_click_script = f'top.frames[\'node_frame\'].location.href=\'{child_node_path}\';'+\
                                    f'top.frames[\'vis_frame\'].location.href=\'{child_node_state_vis_path}\';'
                html += f'<TR><TD class="hvr" onclick="{on_click_script}">{node_html}</TD>'+\
                        f'<TD {p10cl}>{child_node.subtree_max_depth}</TD>'+\
                        f'<TD {p10cl}>{child_node.subtree_nodes_count}</TD>'+\
                        f'<TD {p10cl}></TD></TR>'

            html += '</TABLE></p>\n'

            # Task network
            html += '<hr>'
            html += '<h3>Task network</h3>\n'
            html += '<TABLE><TR><TD><b>id</b></TD><TD><b>class</b></TD>'+\
                                                                '<TD><b>parameters</b></TD></TR>\n'
            task_network = node.task_network
            tn_gr = task_network['ParameterGroundings']
            for task in task_network['Tasks']:
                arg_values: list[str] = []
                colors: list[str] = []
                for arg_name in task['Args']:
                    if arg_name in tn_gr:
                        param_data = tn_gr[arg_name]
                        value_str = self._world_model_module.format_as_text(param_data)
                        arg_values.append(value_str)
                        colors.append('black')
                    else:
                        arg_values.append( '?' )
                        colors.append('red')
                html += f'<TR><TD>{task['Id']}</TD><TD>{task['Class']}</TD>'+\
                            f'<TD>{listToStrTooltip(task['Args'], arg_values, colors)}</TD></TR>\n'
            html += '</TABLE></p>\n'

            html += '<hr>'
            html += '<h3>Generators</h3>\n'

            streams = node.streams
            if not streams:
                generator_instance = None
            else:
                generator_instance = streams[0]['GeneratorInstance']
                html += '<TABLE><TR><TD><b>generator</b></TD><TD><b>inputs</b></TD>'+\
                                                                    '<TD><b>outputs</b></TD></TR>\n'

                for stream in streams:
                    input_values: list[str] = []
                    colors: list[str] = []
                    for arg_name in stream['InputMapping']:
                        if arg_name in tn_gr:
                            param_data = tn_gr[arg_name]
                            value_str = self._world_model_module.format_as_text(param_data)
                            input_values.append(value_str)
                            colors.append('black')
                        else:
                            input_values.append( '?' )
                            colors.append('red')
                    html += f'<TR><TD>{stream['Name']}</TD>'+\
                            f'<TD>{listToStrTooltip(stream['InputMapping'], input_values, colors)}</TD>'+\
                            f'<TD>{listToStr(stream['OutputMapping'])}</TD></TR>\n'
                html += '</TABLE></p>\n'

            # State: predicates
            state = node.state
            html += '<hr>'
            html += '<h3>State: predicates</h3>\n'
            html += '<TABLE><TR><TD><b>name</b></TD><TD><b>parameters</b></TD></TR>\n'
            for pred in state['Predicates']:
                arg_values: list[str] = []
                colors: list[str] = []
                for arg_name in pred['Args']:
                    if arg_name in state['Values']:
                        param_data = state['Values'][arg_name]
                        value_str = self._world_model_module.format_as_text(param_data)
                        arg_values.append(value_str)
                        colors.append('black')
                    else:
                        arg_values.append( '?' )
                        colors.append('red')

                html += f'<TR><TD>{pred['Name']}</TD>'+\
                    f'<TD>{listToStrTooltip(pred['Args'], arg_values, colors)}</TD></TR>\n'
                
            html += '</TABLE></p>\n'

            # Task network parameters
            html += '<hr>'
            html += '<h3>Task network parameters</h3>\n'
            html += '<TABLE><TR><TD><b>name</b></TD><TD><b>type</b></TD>'+\
                                                                '<TD><b>value</b></TD></TR>\n'
            for param_name, param_data in task_network['ParameterGroundings'].items():
                value_str = self._world_model_module.format_as_text(param_data)
                html += f'<TR><TD>{param_name}</TD><TD>{param_data.getType()}</TD>'+\
                                                    f'<TD>{value_str}</TD></TR>\n'
            html += '</TABLE></p>\n'

            # State parameters
            html += '<hr>'
            html += '<h3>State parameters</h3>\n'
            html += '<TABLE><TR><TD><b>name</b></TD><TD><b>type</b></TD>'+\
                                                                    '<TD><b>value</b></TD></TR>\n'
            for param_name, param_data in state['Values'].items():
                if param_data is None:
                    html += f'<TR><TD>{param_name}</TD><TD>{None}</TD>'+\
                                                f'<TD>{None}</TD></TR>\n'
                else:
                    value_str = self._world_model_module.format_as_text(param_data)
                    html += f'<TR><TD>{param_name}</TD><TD>{param_data.getType()}</TD>'+\
                                                f'<TD>{value_str}</TD></TR>\n'
            html += '</TABLE></p>\n'


            #html += '<p><img src="{}.svg" width="{}px" height="{}px" /></p>\n'.format( self.getPath('search_node', ('search_node_dot', node_id)), graph_width, graph_height )
            # if node.getTaskNetwork().isEmpty():
            #     html += '<p>empty</p>\n'
            # else:

            #     if generate_tn_dot:
            #         tn_graph_filename_no_ext = self.getPath('search_node', 'search_node_dot',
            #                                                                         (node_id, ))
            #         tn_graph_filename = '{}.svg'.format(tn_graph_filename_no_ext)
            #         html += '<div class="outsideWrapper">\n'
            #         html += '    <div class="insideWrapper">\n'
            #         html += '        <img id="htn_dot_img" style="border:2px solid red"'\
            #                 ' src="{}" alt="task network" class="coveredImage" />\n'.format(
            #                                                                     tn_graph_filename )
            #         html += '        <canvas id="myCanvas" width="{}px" height="{}px"'.format(graph_width, graph_height)
            #         html += ' style="border:1px solid #d3d3d3;" class="coveringCanvas">Your browser does not support the HTML5 canvas tag.</canvas>\n'
            #         html += '    </div>\n'
            #         html += '</div>\n'


            #     tn_cost = 0.0 #self.__pg.getTaskNetworkCost(h_node_id)
            #     html += '<p>task network cost: {:.2f}</p>\n'.format(tn_cost)

            state = node.state
            html_gen = self._world_model_module.generate_node_visualization_html(
                                state, task_network, generator_instance, self._verbosity_level)
            html_gen = html_gen.replace('<state_info/>', f'state: {state_id}')
            new_dir_path = '{}/cached_data/{}'.format(self.__output_path, state_id)
            # os.makedirs(new_dir_path)
            os.makedirs(new_dir_path+'/vis')
            # Write html page for this particular node
            filepath = self.getPath('abs', 'cached_data_html', (state_id, 'vis'))
            with open( filepath, 'w' ) as f:
                f.write(html_gen)

            # gen_vis_path = self.getPath('search_node', 'cached_data_html', (state_id, 'vis'))
            # stream_name_link = f'<a href="{gen_vis_path}" '+\
            #         f'target="vis_frame">{stream['Name']}</a>\n'
            
            # html += htn_graph_html_script
            html += '</body>\n'
            html += '</html>\n'

            # Write html page for this particular node
            with open( self.getPath('abs', 'search_node_html', (node_id,)), 'w' ) as f:
                f.write(html)

            # Write task data for tasks in task network
            # for task_id in task_id_list:
            #     task = node.getTaskNetwork().getTaskName(task_id)
            #     html = '<!DOCTYPE html>\n'
            #     html += '<html>\n'
            #     html += '<style>\n'
            #     html += '  table, th, td {\n'
            #     html += '    border: 1px solid black;\n'
            #     html += '    border-collapse: collapse;\n'
            #     html += '  }\n'
            #     html += '</style>\n'
            #     # html += '<script type="text/javascript" src="http://static.robotwebtools.org/EventEmitter2/current/eventemitter2.min.js"></script>\n'
            #     # html += '<script type="text/javascript" src="http://static.robotwebtools.org/roslibjs/current/roslib.min.js"></script>\n'
            #     html += '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/eventemitter2@6.4.9/lib/eventemitter2.min.js"></script>\n'
            #     html += '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/roslib@1/build/roslib.min.js"></script>\n'
            #     html += '<script type="text/javascript">\n'
            #     html += self.__getMainLinkScript('search_node')
            #     html += '</script>\n'
            #     js_code_task_vis = None
            #     if not self.__domain_vis is None:
            #         js_code_task_vis, buttons_func = self.__domain_vis.getJsCodeTaskVisualization(
            #                     node.getStateId(), task, node.getTaskNetwork().getTaskArgs(task_id))
            #     if js_code_task_vis:
            #         html += '<script type="text/javascript" src="{}"></script>\n'.format(
            #                 self.getPath('search_node', 'js_init_ros', None))
            #         html += '<script type="text/javascript" src="{}"></script>\n'.format(
            #                 self.getPath('search_node', 'js_init_task_pub', None))
            #         html += '<script type="text/javascript" src="{}"></script>\n'.format(
            #                 self.getPath('search_node', 'js_task_vis', (node_id, task_id[1])))
            #     html += '<body>\n'

            #     html += '<h2>Node {}, Task {}: {}</h2>\n'.format(node_id, task_id[1], task)
            #     html += '<p id="message">Unknown websocket status</p>'

            #     if js_code_task_vis:
            #         for button_name, function_name in buttons_func:
            #             html += '<p><button onclick="{}()">{}</button></p>\n'.format(function_name,
            #                                                                         button_name)
            #         # html += '<p><button onclick="publishTask()">Visualize task</button></p>\n'.format(node_id, task_id[1])
            #     else:
            #         html += '<p>No visualization is available</p>\n'
            #     # TODO:
            #     #if self.__pd.hasVis( task ):
            #     #    html += self.__pd.getTaskDef(task).getVis(self, data_interface, state, node.tn.getTaskArgs(task_id))

            #     html += '</body>\n'
            #     html += '</html>\n'

            #     # Write html page for this particular node
            #     with open( self.getPath('abs', 'task_html', (node_id, task_id[1])), 'w' ) as f:
            #         f.write(html)

            #     if js_code_task_vis:
            #         # Write js code for this particular node
            #         with open( self.getPath('abs', 'js_task_vis', (node_id, task_id[1])), 'w' ) as f:
            #             f.write(js_code_task_vis)

    def __writeFramesPage(self):
        html = '<!DOCTYPE html>\n'
        html += '<html>\n'
        html += '<FRAMESET cols="50%, 50%">\n'
        html += '    <FRAME src="./' + self.getPath('index', 'main_html', None) + '" NAME="node_frame">\n'
        html += '    <FRAME src="" NAME="vis_frame">\n'
        # html += '    <FRAMESET rows="70%, 30%">\n'
        # html += '        <FRAME src="" NAME="vis_frame">\n'
        # html += '        <FRAME src="{}" NAME="graph_frame">\n'.format(self.getPath('index', 'search_graph_html', None))
        # html += '    </FRAMESET>\n'
        html += '</FRAMESET>\n'
        html += '<NOFRAMES>\n'
        html += '<p>HTML frames are not supported!</p>\n'
        html += '</NOFRAMES>\n'
        html += '</html>\n'
        with open(self.__output_path + '/index.html', 'w') as f:
            f.write(html)

        # Write blank page
        html = '<!DOCTYPE html>\n'
        html += '<html>\n'
        html += '<body/>\n'
        html += '</html>\n'
        with open( self.getPath('abs', 'blank_html', None), 'w' ) as f:
            f.write(html)

    def __writeMainPage(self):
        html = '<!DOCTYPE html>\n'
        html += '<html>\n'
        html += '<style>\n'
        html += '  table, th, td {\n'
        html += '    border: 1px solid black;\n'
        html += '    border-collapse: collapse;\n'
        html += '  }\n'
        html += '</style>\n'
        #html += '<script type="text/javascript">\n'
        #html += self.__getMainLinkScript()
        #html += '</script>\n'
        html += '<body>\n'
        html += '<h2>Planning results</h2>\n'
        #html += '<p><a href="{}" target="node_frame" >Data</a></p>\n'.format(
        #            self.getPath('index', 'data_html', None))

        html += '<p><a href="../index.html" target=_parent>Back</a></p>\n'

        html += '<p><a href="{}" target="node_frame" >Planning domain</a></p>\n'.format(
                    self.getPath('index', 'domain_html', None))

        # html += '<p><a href="{}" target="node_frame" >Search tree</a></p>\n'.format(
        #             self.getPath('index', 'search_node_html', (0,)))

        root_node_path = self.getPath('index', 'search_node_html', (0,))
        root_node_state_vis_path = self.getPath('index', 'cached_data_html', (0, 'vis'))
        onclick_script = f'top.frames[\'node_frame\'].location.href=\'{root_node_path}\';'+\
                                f'top.frames[\'vis_frame\'].location.href=\'{root_node_state_vis_path}\';'
        js_load_frames = f'javascript:{{{onclick_script}}}'

        html += f'<p><a href="{js_load_frames}" target="node_frame" >Search tree</a></p>\n'

        if len(self._plans) > 0:
            for idx, plan in enumerate(self._plans):
                solution_path = self.getPath('index', 'solution_html', (idx,))
                # Write plan visualization, if available
                init_state = self._pt.getNode(self._pt.getRootNodeId()).state
                plan_vis_html = self._world_model_module.generate_plan_visualization(init_state, plan)
                if not plan_vis_html is None:
                    with open(self.getPath('abs', 'solution_vis', (idx,)), 'w') as f:
                        f.write(plan_vis_html)
                    solution_vis_path = self.getPath('index', 'solution_vis', (idx,))
                    onclick_script = f'top.frames[\'node_frame\'].location.href=\'{solution_path}\';'+\
                                    f'top.frames[\'vis_frame\'].location.href=\'{solution_vis_path}\';'
                else:
                    onclick_script = f'top.frames[\'node_frame\'].location.href=\'{solution_path}\';'

                js_load_frames = f'javascript:{{{onclick_script}}}'
                # Add link to solution page
                # solution_path = self.getPath('index', 'solution_html', (idx,))
                html += f'<p><a href="{js_load_frames}" target="node_frame" >Solution #{idx} (size: {len(plan)})</a></p>\n'
        else:
            html += '<p>No solutions</p>'

        html += '<p><a href="{}" target="node_frame" >Data</a></p>\n'.format(
                    self.getPath('index', 'data_html', None))

        html += '</body>\n'
        html += '</html>\n'

        # Write html page for this particular node
        with open( self.getPath('abs', 'main_html', None), 'w' ) as f:
            f.write(html)

    def __getMainLinkScript(self, level):
        html = 'function mainClick() {'
        html += 'parent.vis_frame.location=\'{}\';'.format(
                    self.getPath(level, 'blank_html', None) )
        html += '}\n'
        return html

    def __getMainLink(self, level):
        return '<p><a href="{}" onClick="mainClick()" >main</a></p>\n'.format(
                    self.getPath(level, 'main_html', None) )

def listToStr(l):
    assert isinstance(l, list)
    result = ''
    for item in l:
        if result:
            result += ', '
        result += str(item)
    return result

def listToStrTooltip(l_text: list[str], l_tooltip: list[str], l_color: list[str]):
    assert isinstance(l_text, list)
    assert isinstance(l_tooltip, list)
    assert isinstance(l_color, list)
    assert len(l_text) == len(l_tooltip)
    assert len(l_text) == len(l_color)
    result = ''
    for txt, tooltip, color in zip(l_text, l_tooltip, l_color):
        if result:
            result += ', '
        result += f'<span style="color:{color}" title="{tooltip}">{txt}</span>'
    return result
