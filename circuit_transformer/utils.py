# Author: Xihan Li
# xihan.li@cs.ucl.ac.uk
# https://snowkylin.github.io
#
# Implementation of "Circuit Transformer: A Transformer That Preserves Logical Equivalence"
# https://openreview.net/forum?id=kpnW12Lm9p

from __future__ import annotations

import copy
import json as json  # Legacy public utility exports.
import typing
import npn
import numpy as np
import os
import graphviz
import tempfile
import itertools as itertools
import time
import platform
import subprocess
import shutil
import uuid as uuid
import pickle as pickle
import multiprocessing
import queue
from functools import partial
import bitarray
import bitarray.util

__all__ = [
    'copy', 'json', 'typing', 'npn', 'np', 'os', 'graphviz', 'tempfile',
    'itertools', 'time', 'platform', 'subprocess', 'uuid', 'pickle',
    'multiprocessing', 'bitarray', 'base_path', 'abc_path', 'aigtoaig_path',
    'temp_dir', 'temp_dir_init', 'Node', 'NodeWithInv', 'plot_network',
    'compute_input_tt', 'compute_tt', 'compute_tts', 'check_conflict',
    'sequential_synthesis', 'structural_hashing', 'cec', 'read_aiger',
    'write_aiger', 'get_aig_info', 'npn_transform', 'npn_transform_tt',
    'tt_to_str', 'count_num_ands', 'count_seq_length', 'check_integrity',
    'base_2_log', 'get_inputs_rec', 'popcount', 'reverse_permutation',
    'NPNP_transformation', 'detect_circle', 'get_subcircuit_nodes',
    'aag_to_aig', 'divide_chunks', 'gpu_count', 'MPDataset', 'LogWrapper',
]


base_path = os.path.abspath(os.path.dirname(__file__))
abc_path = base_path + ("/bin/abc" if platform.system() == 'Linux' else "bin\\abc")
aigtoaig_path = base_path + ("/bin/aigtoaig" if platform.system() == 'Linux' else "bin\\aigtoaig")
temp_dir = None


def temp_dir_init():
    global temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(dir=temp_dir)
        # print("aig file saved to %s" % temp_dir)
    return temp_dir


class Node:
    def __init__(self, var: int | None, left: NodeWithInv | None, right: NodeWithInv | None):
        self.var = var      # var >= 0: x_{var}; var = -1: constant 0; var = -2: all possibilities of input nodes (x_0, not(x_0), x_1, not(x_1), ...)
        self.left = left
        self.right = right
        self._input_symbol: str | None = None

    @property
    def input_symbol(self) -> str | None:
        return self._input_symbol

    @input_symbol.setter
    def input_symbol(self, value: str | None):
        self._input_symbol = value

    def is_leaf(self):
        return self.var is not None

    def __lt__(self, other):    # for sorting a list of nodes
        return id(self) < id(other)


class NodeWithInv(Node):
    def __init__(self, parent: Node, inverted, output_symbol=None):
        self._parent = parent
        self.inverted = inverted
        self.output_symbol = output_symbol

    @property
    def var(self):
        return self._parent.var

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        assert type(value) is Node
        self._parent = value

    @var.setter
    def var(self, value):
        self._parent.var = value

    @property
    def left(self):
        return self._parent.left

    @left.setter
    def left(self, value):
        self._parent.left = value

    @property
    def right(self):
        return self._parent.right

    @right.setter
    def right(self, value):
        self._parent.right = value

    @property
    def input_symbol(self):
        return self._parent.input_symbol

    @input_symbol.setter
    def input_symbol(self, value):
        self._parent.input_symbol = value


def plot_network(roots: NodeWithInv | Node | list[NodeWithInv | Node], view=False, tree=False, filename: str | None = None,
                 title: str | None = None, show_address=False, highlight_nodes: set | None = None):
    def plot_network_rec(node: NodeWithInv | None, parent: NodeWithInv | None, depth: int, g: graphviz.Digraph, plotted_nodes: set):
        # name = "x_%d" % node.var if node.var is not None else repr(node)
        if node is None:
            return
        if tree:
            node = copy.copy(node)                  # NodeWithInv
            node.parent = copy.copy(node.parent)    # Node
            new_nodes.append(node)  # save the node instance so that it won't be destroyed (to avoid id reallocation)
        name = str(id(node.parent))
        if node.var is not None:
            g.attr('node', color='lightyellow' if highlight_nodes is None or node.parent not in highlight_nodes else 'yellow')
            g.node(name, (("x_%d" % node.var) if node.var >= 0 else "0" if node.var == -1 else "not determined") + ("\n" + hex(id(node.parent)) if show_address else ""))
            g.attr('node', color='lightgrey')
        else:
            g.attr('node', color='lightgrey' if highlight_nodes is None or node.parent not in highlight_nodes else 'yellow')
            g.node(name, "AND" + ("\n" + hex(id(node.parent)) if show_address else ""))
            g.attr('node', color='lightgrey')
            if node.parent not in plotted_nodes:
                plot_network_rec(node.left, node, depth + 1, g, plotted_nodes)
                plot_network_rec(node.right, node, depth + 1, g, plotted_nodes)
                plotted_nodes.add(node.parent)
        if parent is not None:
            if node.inverted:
                g.attr('edge', style='dashed')
            g.edge(name, str(id(parent.parent)))
            if node.inverted:
                g.attr('edge', style='')

    new_nodes = []
    g = graphviz.Digraph('logic_network' if title is None else title)

    g.attr('node', style='filled')
    if not isinstance(roots, list):
        roots = [roots]
    output_roots = [root if isinstance(root, NodeWithInv) else NodeWithInv(root, inverted=False) for root in roots]
    plotted_nodes = set()
    g.attr('node', color='lightgrey')
    for root in output_roots:
        plot_network_rec(root, None, 0, g, plotted_nodes)
    g.attr('node', color='lightblue')
    for i, root in enumerate(output_roots):
        name = str(id(root.parent)) + str(root.inverted)
        g.node(name, "o_%d" % i)
        if root.inverted:
            g.attr('edge', style='dashed')
        g.edge(str(id(root.parent)), name)
        if root.inverted:
            g.attr('edge', style='')
    if filename is not None:
        return g.render(outfile=filename, view=view)
    else:
        return g.render(directory=tempfile.mkdtemp(), view=view)


def compute_input_tt(num_inputs):
    input_tt = [bitarray.bitarray(2 ** num_inputs) for _ in range(num_inputs * 2)]
    for i in range(num_inputs):
        for j in range(2 ** num_inputs):
            input_tt[i * 2][j] = (j >> i) % 2
            input_tt[i * 2 + 1][j] = not input_tt[i * 2][j]
    return input_tt


def compute_tt(root: NodeWithInv | Node, num_inputs=None, input_tt=None, cache=None):
    if input_tt is None:
        if num_inputs is None:
            raise ValueError('Provide num_inputs or input_tt.')
        input_tt = compute_input_tt(num_inputs)
    elif num_inputs is None:
        num_inputs = base_2_log(len(input_tt[0]))
    truth_tables = input_tt
    table_size = 2 ** num_inputs

    def compute_tt_rec(root: NodeWithInv | None) -> bitarray.bitarray:
        if isinstance(cache, dict) and root in cache:
            return cache[root]
        if root is None:
            raise ValueError('root is None')
        elif root.var is not None:
            if root.var >= 0:
                res = truth_tables[root.var * 2]
            elif root.var == -1:    # -1 means constant zero
                res = bitarray.util.zeros(table_size)
            else:                   # -2 means not decided
                raise ValueError('root.value = %d, not supported' % root.var)
        elif isinstance(cache, dict) and root.parent in cache:
            res = cache[root.parent]
        else:
            left_tt = compute_tt_rec(root.left)
            right_tt = compute_tt_rec(root.right)
            res = left_tt & right_tt
        if root.inverted:
            res = ~res
        return res

    wrapped_root = root if isinstance(root, NodeWithInv) else NodeWithInv(root, inverted=False)
    res = compute_tt_rec(wrapped_root)
    return res


def compute_tts(roots: NodeWithInv | list[NodeWithInv], num_inputs=None, input_tt=None):
    if isinstance(roots, NodeWithInv):
        roots = [roots]
    if input_tt is None:
        assert num_inputs is not None
        input_tt = compute_input_tt(num_inputs)
    return [compute_tt(root, input_tt=input_tt) for root in roots]


def check_conflict(tree_stack: list[NodeWithInv], tt, input_tt, cache):
    tt_size = len(tt)
    b = input_tt.copy()
    n = [bitarray.util.zeros(tt_size) for _ in input_tt]   # whether is unknown
    for node in reversed(tree_stack):
        if node.right is None:
            for i in range(len(input_tt)):
                n[i] = b[i] | n[i]              # set n[i] to True for those b[i] is True (1 AND U = U)
        else:
            left_tt = cache[node.left]          # all the left child have a known truth table
            for i in range(len(input_tt)):
                b[i] = b[i] & left_tt
                n[i] = n[i] & left_tt                     # set n[i] to False for those left_tt[i] is False (0 AND U = 0)
        if node.inverted:
            for i in range(len(input_tt)):
                b[i] = ~b[i]
    has_conflict = bitarray.bitarray([((~n[i]) & (b[i] ^ tt)).any() for i in range(len(input_tt))])   # any n[i] = False and b[i] != tt[i]
    completeness = bitarray.bitarray([not n[i].any() for i in range(len(input_tt))])    # whether all the outputs are known
    return has_conflict, completeness


def _synthesize(roots, command, verbose=False):
    commands = {
        'resyn2': 'read input.aig; resyn2; write result.aig;',
        'deepsyn': '&read input.aig; &deepsyn; &write -n result.aig;',
        'transtoch': '&read input.aig; &transtoch -M 1 -V 0; &write -n result.aig;',
        'strash': 'read input.aig; strash; write result.aig;',
    }
    if command not in commands:
        raise ValueError(f'Unsupported synthesis command: {command}')
    # Fixed filenames in a private working directory avoid shell quoting errors
    # and collisions between concurrent synthesis calls.
    with tempfile.TemporaryDirectory(dir=temp_dir_init()) as directory:
        shutil.copyfile(os.path.join(base_path, 'bin', 'abc.rc'),
                        os.path.join(directory, 'abc.rc'))
        write_aiger(roots, os.path.join(directory, 'input.aig'), with_symbol_table=False)
        result = subprocess.run([abc_path, '-c' if verbose else '-q', commands[command]],
                                cwd=directory, capture_output=not verbose, text=True, check=True)
        output = os.path.join(directory, 'result.aig')
        if not os.path.isfile(output):
            raise RuntimeError(f'ABC did not produce a circuit: {result.stdout or ""} {result.stderr or ""}')
        return read_aiger(output)


def sequential_synthesis(roots, verbose=False, command='resyn2', title: str = "sequential"):
    return _synthesize(roots, command, verbose)


def structural_hashing(roots, verbose=False):
    return _synthesize(roots, 'strash', verbose)


def cec(roots_1, roots_2, verbose=0):
    with tempfile.TemporaryDirectory(dir=temp_dir_init()) as directory:
        for index, roots in enumerate((roots_1, roots_2)):
            filename = os.path.join(directory, f'input{index}.aig')
            if isinstance(roots, (str, os.PathLike)):
                shutil.copyfile(roots, filename)
            else:
                write_aiger(roots, filename, with_symbol_table=False)
        result = subprocess.run([abc_path, '-c', 'cec input0.aig input1.aig'],
                                cwd=directory, capture_output=True, text=True, check=True)
        stdout = result.stdout
        equivalent = 'Networks are equivalent' in stdout
        if not equivalent and 'Networks are NOT EQUIVALENT' not in stdout:
            raise RuntimeError(f'ABC equivalence check failed: {stdout} {result.stderr}')
        if verbose > 0 or not equivalent:
            print(stdout)
        return equivalent


def read_aiger(filename=None, aiger_str: str | None = None) -> tuple[NodeWithInv | list[NodeWithInv], list[int]]:
    if filename is not None:
        filename_without_ext, file_ext = os.path.splitext(filename)
        if file_ext == '.aig':  # binary
            subprocess.run([aigtoaig_path, os.fspath(filename), filename_without_ext + '.aag'], check=True)
        filename = filename_without_ext + ".aag"
        with open(filename) as f:
            aiger_str = f.read()
    if aiger_str is None:
        raise ValueError('Provide filename or aiger_str.')
    f = aiger_str.splitlines()
    lineid = 0
    s = f[lineid]
    lineid += 1
    info = list(map(int, s.split(' ')[1:]))
    max_var_index, num_inputs, num_latches, num_outputs, num_ands = info
    assert num_latches == 0    # a boolean network without latches
    node_dict = {i: Node(None, None, None) for i in range(0, max_var_index + 1)}
    node_dict[0].var = -1   # constant zero
    input_indices, output_indices, output_not_list = [], [], []
    output_symbol_dict = {}
    for i in range(num_inputs):
        s = int(f[lineid])
        lineid += 1
        var_index = s // 2
        if var_index <= 0:
            print(aiger_str)
            raise ValueError()
        input_indices.append(var_index)
        node_dict[var_index].var = var_index - 1
        node_dict[var_index].left = None
        node_dict[var_index].right = None
    for i in range(num_outputs):
        s = int(f[lineid])
        lineid += 1
        output_indices.append(s // 2)
        output_not_list.append(s % 2 == 1)
    for i in range(num_ands):
        s, p, q = list(map(int, f[lineid].split(' ')))
        lineid += 1
        var_index = s // 2
        left_not = p % 2 == 1
        if left_not:
            left = NodeWithInv(node_dict[p // 2], True)
        else:
            left = NodeWithInv(node_dict[p // 2], False)
        right_not = q % 2 == 1
        if right_not:
            right = NodeWithInv(node_dict[q // 2], True)
        else:
            right = NodeWithInv(node_dict[q // 2], False)
        node_dict[var_index].var = None
        node_dict[var_index].left = left
        node_dict[var_index].right = right
    while lineid < len(f) and f[lineid] not in ('c', ''):
        id, symbol = f[lineid].split(' ', 1)
        ioro, id = id[0], int(id[1:])
        if ioro == 'i':
            node_dict[input_indices[id]].input_symbol = symbol
        elif ioro == 'o':
            output_symbol_dict[id] = symbol
        lineid += 1
    output_nodes = []
    for i, (output_index, output_not) in enumerate(zip(output_indices, output_not_list)):
        output_nodes.append(NodeWithInv(node_dict[output_index], output_not,
                                        output_symbol_dict[i] if i in output_symbol_dict else None))
    if num_outputs == 1:    # boolean function
        return output_nodes[0], info
    return output_nodes, info


def write_aiger(root: NodeWithInv | list[NodeWithInv], filename: str | None = None, with_symbol_table=True, num_inputs=None) -> str:
    def write_aiger_inputs_rec(root: NodeWithInv | None, info: dict, visited_nodes: set):
        if root is None:
            raise ValueError('Cannot serialize an incomplete circuit.')
        if root.var is not None:
            if root.var == -1:
                info['node_to_id'][root.parent] = 0
                return
            if root.var < 0:
                raise ValueError('Cannot serialize an undetermined input.')
            # root.id = root.var + 1  # id starts from 1
            info['node_to_id'][root.parent] = root.var + 1
            info['inputs'][root.var + 1] = root.input_symbol
        else:
            if root not in visited_nodes:
                write_aiger_inputs_rec(root.left, info, visited_nodes)
                write_aiger_inputs_rec(root.right, info, visited_nodes)
                visited_nodes.add(root)

    def write_aiger_rec(root: NodeWithInv | None, info: dict):
        if root is None:
            raise ValueError('Cannot serialize an incomplete circuit.')
        if not root.is_leaf():
            if root.parent not in info['node_to_id']:
                left_id, left_not = write_aiger_rec(root.left, info)
                right_id, right_not = write_aiger_rec(root.right, info)
                info['node_to_id'][root.parent] = info['node_id']
                info['ands_detail'].append((info['node_id'], left_id * 2 + left_not, right_id * 2 + right_not))
                info['node_id'] += 1
        # assert root.id is not None
        return info['node_to_id'][root.parent], root.inverted

    info = {'node_id': 1, 'ands_detail': [], 'inputs': dict(), 'node_to_id': dict()}
    if isinstance(root, NodeWithInv):
        root = [root]
    root = root.copy()
    for i, r in enumerate(root):
        if type(r) is Node:
            root[i] = NodeWithInv(r, inverted=False)
    for r in root:
        write_aiger_inputs_rec(r, info, set())
    if num_inputs is None:
        info['node_id'] = max(info['inputs'].keys(), default=0) + 1   # skip the input ids
    else:
        if max(info['inputs'].keys(), default=0) > num_inputs:
            print("Warning: num_inputs is smaller than the actual number of inputs, use the latter instead")
            num_inputs = max(info['inputs'].keys())
        info['node_id'] = num_inputs + 1    # input ids from 1 to num_inputs
    outputs = []
    for r in root:
        output_id, output_not = write_aiger_rec(r, info)
        outputs.append(output_id * 2 + output_not)
    info['node_id'] -= 1
    aiger_str = "aag %d %d 0 %d %d\n" % \
                (info['node_id'], len(info['inputs']), len(outputs), len(info['ands_detail']))
    for i in sorted(info['inputs'].keys()):
        aiger_str += str(i * 2) + '\n'
    for i in outputs:
        aiger_str += str(i) + '\n'
    for and_id, left_literal, right_literal in info['ands_detail']:
        aiger_str += "%d %d %d\n" % (and_id * 2, left_literal, right_literal)
    if with_symbol_table:
        for i, (id, symbol) in enumerate(sorted(info['inputs'].items(), key=lambda x: x[0])):
            if symbol is not None:
                aiger_str += 'i%d %s\n' % (i, symbol)
        # output_used_symbols = set()
        for i, r in enumerate(root):
            if r.output_symbol is not None:
                aiger_str += 'o%d %s\n' % (i, r.output_symbol)
            # if r.output_symbol is not None:
            #     for s in r.output_symbol:
            #         if s not in output_used_symbols:
            #             selected_s = s
            #             break
            #     aiger_str += 'o%d %s\n' % (i, selected_s)
            #     output_used_symbols.add(selected_s)
    if filename is not None:
        filename_without_ext, file_ext = os.path.splitext(filename)
        with open(filename_without_ext + ".aag", 'w', newline='\n') as f:
            f.write(aiger_str)
        if file_ext == '.aig':     # binary
            subprocess.run([aigtoaig_path, filename_without_ext + '.aag',
                            filename_without_ext + '.aig'], check=True)
            # os.unlink(filename_without_ext + ".aag")
    return aiger_str


def get_aig_info(filename):
    if not os.path.exists(filename):
        return [-1, -1, -1, -1, -1]
    with open(filename, 'rb') as f:
        s = f.readline().decode('ascii')
        if not s:
            raise ValueError('Empty AIGER file.')
        info = list(map(int, s.split(' ')[1:]))
    return info


def npn_transform(roots: NodeWithInv | list[NodeWithInv], phase, perm,
                  output_invs: bool | list[bool] | np.ndarray):
    def n_transform(root: NodeWithInv | None, phase, perm, transformed_node: set, transformed_node_with_inv: set):
        if root is None:
            raise ValueError('Cannot transform an incomplete circuit.')
        if root.var is not None:
            if root.var < 0:
                return
            if root not in transformed_node_with_inv:
                if phase[root.var]:
                    root.inverted = not root.inverted
                transformed_node_with_inv.add(root)
            if root.parent not in transformed_node:
                # root.var = perm[root.var]
                transformed_node.add(root.parent)
        else:
            n_transform(root.left, phase, perm, transformed_node, transformed_node_with_inv)
            n_transform(root.right, phase, perm, transformed_node, transformed_node_with_inv)

    leaf_nodes = set()
    transformed_node_with_inv = set()
    if isinstance(roots, NodeWithInv):
        roots = [roots]
    for root in roots:
        n_transform(root, phase, perm, leaf_nodes, transformed_node_with_inv)
    for node in leaf_nodes:
        node.var = perm[node.var]
    if isinstance(output_invs, bool):
        output_invs = [output_invs]
    for output_inv, root in zip(output_invs, roots):
        if output_inv:
            root.inverted = not root.inverted
    return roots if len(roots) > 1 else roots[0]


def npn_transform_tt(tt: bitarray.bitarray, phase, perm, output_invs: bool | list[bool]):
    return bitarray.bitarray(npn.transform_tt(tt.tolist(), phase, perm, output_invs))


def tt_to_str(tt, hex=False):
    if hex:
        return ("%0" + str(len(tt) // 4) + "x") % npn.tt_to_int(tt)
    else:
        return "".join("1" if i else "0" for i in tt)


def count_num_ands(root: NodeWithInv | list | None, node_set: set | None = None):
    if node_set is None:
        node_set = set()
    if isinstance(root, list):
        return sum([count_num_ands(r, node_set) for r in root])
    if root is None or root.var is not None or root.parent in node_set:
        return 0
    else:
        node_set.add(root.parent)
        return count_num_ands(root.left, node_set) + count_num_ands(root.right, node_set) + 1


def count_seq_length(root: NodeWithInv | list[NodeWithInv] | Node | list[Node] | None,
                     maximum_length=None, cut=None, cumulative_length=0):
    if isinstance(root, list):
        for r in root:
            if type(r) is Node:
                r = NodeWithInv(r, inverted=False)
            cumulative_length = count_seq_length(r, maximum_length, cut, cumulative_length)
            if cumulative_length == -1 or (maximum_length is not None and cumulative_length > maximum_length):
                return -1
        return cumulative_length
    if root is not None and not isinstance(root, NodeWithInv):
        root = NodeWithInv(root, inverted=False)
    if root is None or root.is_leaf() or (cut is not None and root.parent in cut):
        return cumulative_length + 1
    else:
        cumulative_length = count_seq_length(root.left, maximum_length, cut, cumulative_length)
        if cumulative_length == -1 or (maximum_length is not None and cumulative_length > maximum_length):
            return -1
        cumulative_length = count_seq_length(root.right, maximum_length, cut, cumulative_length)
        if cumulative_length == -1 or (maximum_length is not None and cumulative_length + 1 > maximum_length):
            return -1
        return cumulative_length + 1


def check_integrity(root: NodeWithInv | list[NodeWithInv] | None):
    if isinstance(root, list):
        return all([check_integrity(r) for r in root])
    if root is None:
        return False
    elif root.is_leaf():
        return True
    else:
        return check_integrity(root.left) and check_integrity(root.right)


def base_2_log(n: int):     # see src/misc/util/abc_global.h in abc
    if n < 2:
        return n
    else:
        r = 0
        n -= 1
        while n > 0:
            r += 1
            n >>= 1
        return r


def get_inputs_rec(node: NodeWithInv | list[NodeWithInv] | None, visited=None, inputs_dict=None):
    if node is None:
        raise ValueError('Cannot traverse an incomplete circuit.')
    if visited is None:
        visited = set()
    if inputs_dict is None:
        inputs_dict = dict()
    if isinstance(node, list):
        for node_i in node:
            get_inputs_rec(node_i, visited, inputs_dict)
        return inputs_dict
    if node in visited:
        return inputs_dict
    visited.add(node)
    if node.is_leaf():
        if node.var not in inputs_dict:
            inputs_dict[node.var] = set()
        inputs_dict[node.var].add(node)
    else:
        get_inputs_rec(node.left, visited, inputs_dict)
        get_inputs_rec(node.right, visited, inputs_dict)
    return inputs_dict


def popcount(tt, var_id: int):
    popcount_0, popcount_1 = 0, 0
    interval = 1 << var_id
    for i, t in enumerate(tt):
        if t:
            if i & interval == 0:
                popcount_0 += 1
            else:
                popcount_1 += 1
    return popcount_0, popcount_1


def reverse_permutation(perm: list):
    reversed_perm = copy.copy(perm)
    for i, p in enumerate(perm):
        reversed_perm[p] = i
    return reversed_perm


def NPNP_transformation(nodes: list[NodeWithInv], npnp_info: list):
    new_nodes = copy.deepcopy(nodes)
    input_flip, input_perm, output_flip, output_perm = npnp_info
    # output perm
    new_nodes = [new_nodes[i] for i in reverse_permutation(output_perm)]
    # input flip
    inputs_dict = get_inputs_rec(new_nodes)
    for i, f in enumerate(input_flip):
        if f and i in inputs_dict:
            for node_with_inv in inputs_dict[i]:
                node_with_inv.inverted = not node_with_inv.inverted
    # input perm
    for i, p in enumerate(reverse_permutation(input_perm)):
        if p in inputs_dict:
            node = list(inputs_dict[p])[0].parent
            node.var = i
    # output flip
    for new_node, f in zip(new_nodes, output_flip):
        if f:
            new_node.inverted = not new_node.inverted
    return new_nodes


def detect_circle(root: NodeWithInv | list[NodeWithInv] | None, visiting=None, visited=None):
    if root is None:
        return False
    if visiting is None or visited is None:
        visiting, visited = set(), set()
    if isinstance(root, list):
        return any([detect_circle(r, visiting, visited) for r in root])
    if root.parent in visiting and root.parent not in visited:
        return True
    if root.parent in visited:
        return False
    visiting.add(root.parent)
    if not root.is_leaf():
        res = detect_circle(root.left, visiting, visited) or detect_circle(root.right, visiting, visited)
    else:
        res = False
    visited.add(root.parent)
    return res


def get_subcircuit_nodes(inputs, outputs):
    def get_subcircuit_nodes_rec(root: NodeWithInv | None, inputs, node_set):
        if root is None:
            raise ValueError('Cannot traverse an incomplete circuit.')
        if root.parent not in node_set:
            node_set.add(root.parent)
            if not root.is_leaf() and root.parent not in inputs:
                get_subcircuit_nodes_rec(root.left, inputs, node_set)
                get_subcircuit_nodes_rec(root.right, inputs, node_set)
    node_set = set()
    for output in outputs:
        if type(output) is Node:
            output = NodeWithInv(output, False)
        get_subcircuit_nodes_rec(output, inputs, node_set)
    return node_set


def aag_to_aig(aag_path, aig_path):
    subprocess.run([aigtoaig_path, os.fspath(aag_path), os.fspath(aig_path)], check=True)


# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# https://pypi.org/project/nvidia-ml-py/
def gpu_count():
    from pynvml import nvmlInit, nvmlDeviceGetCount
    nvmlInit()
    return nvmlDeviceGetCount()


class MPDataset:
    def __init__(self,
                 data: list,
                 mapping_func: typing.Callable,
                 kwargs: dict | None = None,
                 num_processes=4,
                 validation_split=0.1):
        self.train_data, self.validation_data = \
            data[: int(len(data) * (1 - validation_split))], data[int(len(data) * (1 - validation_split)):]
        if num_processes < 0 or not 0 <= validation_split < 1:
            raise ValueError('Expected num_processes >= 0 and validation_split in [0, 1).')
        self.mapping_func = mapping_func if kwargs is None else partial(mapping_func, **kwargs)
        self.num_processes = num_processes
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        # self.pool = multiprocessing.Pool(num_processes)
        self.chunk_size = 128 * 8
        # self.mapped_data_dir = tempfile.mkdtemp()
        #
        def run(num_processes, mapping_func, train_data, chunk_size, q, stop_event):
            print("process start running")
            # Do not wait for queued prefetched batches to drain on shutdown.
            q.cancel_join_thread()
            with multiprocessing.Pool(num_processes) as pool:
                chunk_id = 0
                try:
                    while train_data and not stop_event.is_set():
                        if q.qsize() < chunk_size:
                            chunk = train_data[chunk_id * chunk_size: (chunk_id + 1) * chunk_size]
                            for mapped_data in pool.imap_unordered(mapping_func, chunk, chunksize=128):
                                if isinstance(mapped_data, Exception):
                                    raise mapped_data
                                q.put(mapped_data)
                            chunk_id += 1
                            if chunk_id * chunk_size >= len(train_data):
                                chunk_id = 0
                        else:
                            stop_event.wait(0.1)
                except Exception as exc:
                    q.put(RuntimeError(f'Dataset worker failed: {type(exc).__name__}: {exc}'))
                    # Keep the queue alive until the consumer receives the error.
                    stop_event.wait()

        if self.num_processes > 1:
            self.process = multiprocessing.Process(
                target=run,
                args=(num_processes, self.mapping_func, self.train_data, self.chunk_size,
                      self.queue, self.stop_event)
            )
            self.process.start()

    def __len__(self):
        return len(self.train_data)

    def train_generator(self):
        if self.num_processes > 1:
            # Preserve N-1 examples and queue carryover across epochs.
            for _ in range(max(len(self.train_data) - 1, 0)):
                while True:
                    try:
                        item = self.queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        if not self.process.is_alive():
                            raise RuntimeError('Dataset producer exited before returning all examples.')
                if isinstance(item, Exception):
                    raise item
                yield item
        else:
            for data in self.train_data:
                yield self.mapping_func(data)

    def close(self):
        self.stop_event.set()
        if self.num_processes > 1:
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
        self.queue.close()

    def validation_generator(self):
        for data in self.validation_data:
            mapped_data = self.mapping_func(data)
            if isinstance(mapped_data, Exception):
                raise mapped_data
            yield mapped_data

    def benchmark(self, num_data=512):
        counter = 0
        start_time = time.time()
        for data in self.train_generator():
            counter += 1
            if counter >= num_data:
                break
        total_time = time.time() - start_time
        print("%d data elements processed in %.2fs, average %.4fs per data element" %
              (num_data, total_time, total_time / num_data))


class LogWrapper(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()
