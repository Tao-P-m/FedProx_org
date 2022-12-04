import numpy as np

col, row, thres = 1, 8, 4
terre = [[6, 5, 4, 6, 3, 2, 0, 2]]

col, row, thres = 4, 8, 4
terre = [[5, 4, 5, 2, 5, 3, 9, 6],
        [4, 2, 6, 5, 6, 2, 6, 5],
        [3, 2, 5, 5, 5, 7, 2, 5],
        [9, 4, 7, 5, 6, 7, 1, 5]]

def main():
    if col > row or col < 1:
        return
    if np.max(terre) - np.min(terre) <= thres:
        print(int(col * row))
        return
    
    # find peaks of each col (could be many)
    if col == 1:
        alter = np.array(terre)-np.max(terre)
        peaks = np.where(alter.squeeze() == 0)[0]
        print("index of peak", peaks)
        area_candidate = []
        for idx in peaks:
            # to left
            if idx == 0: li = idx
            if idx > 0:
                li = idx - 1
                while li >= 0:
                    if terre[0][idx] - terre[0][li] > thres:    break
                    else:   li -= 1
            # to right
            if idx == row - 1: ri = row - 1
            if idx < row-1:
                ri = idx + 1
                while ri < row:
                    if terre[0][idx] - terre[0][ri] > thres:    break
                    else:   ri += 1
            #print(li, idx, ri)
            area_candidate.append(ri-li-1)      
        print("area: {}".format(max(area_candidate)))
        return
    
    col_candidate = []
    for c in range(col):
        ter = terre[c]
        alter = np.array(ter)-np.max(ter)
        peaks = np.where(alter.squeeze() == 0)[0]
        print("index of peak", [c, peaks]) # coor of possible elevation
        area_candidate = []
        for idx in peaks:
            # to left - most 
            if idx == 0: li = idx
            if idx > 0:
                li = idx - 1
                while li >= 0:
                    if terre[c][idx] - terre[c][li] > thres:    break
                    else:   li -= 1
            # to right - most
            if idx == row - 1: ri = row - 1
            if idx < row-1:
                ri = idx + 1
                while ri < row:
                    if terre[c][idx] - terre[c][ri] > thres:    break
                    else:   ri += 1
            # to top - most 
            if c == 0:  ti = 0
            if c > 0:
                ti = c - 1
                while ti >= 0:
                    if terre[c][idx] - terre[ti][idx] > thres:  break
                    else:   ti -= 1
            # to bottom - most 
            if c == col - 1:    bi = 0
            if c < col - 1:
                bi = c + 1
                while bi < col:
                    if terre[c][idx] - terre[bi][idx] > thres:  break
                    else:   bi += 1
            #print(li, idx, ri)
            area_candidate.append((ri-li-1)*(bi-ti-1))
        print("area candidates", area_candidate)
        col_candidate.append(max(area_candidate))
    return 

if __name__ == '__main__':
    main()