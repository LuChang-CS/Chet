import numpy as np

from preprocess.parse_csv import EHRParser


def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01):
    print('generating code code adjacent matrix ...')
    n = code_num
    adj = np.zeros((n, n), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]:
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1
                    adj[c_j, c_i] += 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    norm_adj = normalize_adj(adj)
    a = norm_adj < threshold
    b = adj.sum(axis=-1, keepdims=True) > (1 / threshold)
    adj[np.logical_and(a, b)] = 0
    return adj


def normalize_adj(adj):
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result


def generate_neighbors(code_x, lens, adj):
    n = len(code_x)
    neighbors = np.zeros_like(code_x, dtype=bool)
    # a = 0
    # b = 100000
    # c = -1
    # nn = 0
    for i, admissions in enumerate(code_x):
        print('\r\t%d / %d' % (i + 1, n), end='')
        for j in range(lens[i]):
            codes_set = set(np.where(admissions[j] == 1)[0])
            all_neighbors = set()
            for code in codes_set:
                code_neighbors = set(np.where(adj[code] > 0)[0]).difference(codes_set)
                all_neighbors.update(code_neighbors)
            if len(all_neighbors) > 0:
                neighbors[i, j, np.array(list(all_neighbors))] = 1
            # a += len(all_neighbors)
            # if b > len(all_neighbors):
            #     b = len(all_neighbors)
            # if c < len(all_neighbors):
            #     c = len(all_neighbors)
            # nn += 1
    print('\r\t%d / %d' % (n, n))
    # print(b, c, a / nn);exit()
    return neighbors


def divide_middle(code_x, neighbors, lens):
    n = len(code_x)
    divided = np.zeros((*code_x.shape, 3), dtype=bool)
    for i, admissions in enumerate(code_x):
        print('\r\t%d / %d' % (i + 1, n), end='')
        divided[i, 0, :, 0] = admissions[0]
        for j in range(1, lens[i]):
            codes_set = set(np.where(admissions[j] == 1)[0])
            m_set = set(np.where(admissions[j - 1] == 1)[0])
            n_set = set(np.where(neighbors[i][j - 1] == 1)[0])
            m1 = codes_set.intersection(m_set)
            m2 = codes_set.intersection(n_set)
            m3 = codes_set.difference(m_set).difference(n_set)
            if len(m1) > 0:
                divided[i, j, np.array(list(m1)), 0] = 1
            if len(m2) > 0:
                divided[i, j, np.array(list(m2)), 1] = 1
            if len(m3) > 0:
                divided[i, j, np.array(list(m3)), 2] = 1
    print('\r\t%d / %d' % (n, n))
    return divided


def parse_icd9_range(range_: str) -> (str, str, int, int):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    import os
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix
