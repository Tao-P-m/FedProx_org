import numpy as np
from scipy.special import comb, perm
from itertools import combinations, permutations

def decomposition(num):
    decom = []
    while num != 1:
        for i in range(2, num+1):
            if num % i == 0:
                decom.append(i)
                num = num // i
                break
    return decom

def good_palindrome(n, m):
    if n == 2:
        return m
    if n == 4:
        return int(comb(m, 2) * (2*2 + 2))
    

def main():
    N = 4
    M = 10
    if N % 2 == 1 or N > 5 * 10**5:
        return
    if M < 1 or M >= 998244353:
        return
    if M > 9:
        M = 9
    ## list all possible seqs from minimum sub-seqs
    decom = decomposition(N)
    if len(set(decom)) == 1:
        valid_set_size = good_palindrome(N, M)
        print(valid_set_size)
    else:
        pass
   
    subset = set(list(combinations(list(range(1,M+1))*N, N)))
    
    return

if __name__ == "__main__":
    main()