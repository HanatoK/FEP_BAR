#!/usr/bin/env python3
import numpy as np
import argparse

def ensemblesFromOutput(outputfile):
    ensembles = []
    with open(outputfile, 'r') as fInput:
        ensemble = []
        start_collecting = False
        for line in fInput:
            if line.startswith('#NEW FEP WINDOW'):
                ensemble = []
            if line.startswith('FepEnergy:') and start_collecting:
                fields = line.split()
                ensemble.append(float(fields[6]))
            if line.startswith('#STARTING COLLECTION'):
                start_collecting = True
            if line.startswith('#Free energy change'):
                start_collecting = False
                if ensemble:
                    ensembles.append(np.array(ensemble))
    return ensembles

# debug the data reading
def showEnsembles(ensembles, title):
    print(title)
    print('=' * 80)
    print(f'Number of windows: {len(ensembles)}')
    for i, ensemble in enumerate(ensembles):
        print(f'Samples in window {i:3d}: {len(ensemble)}')
    print('=' * 80)
    print('\n')


def Fermi(x):
    return 1.0 / (1.0 + np.exp(x))

def estimateDeltaA(forward_deltaU, backward_deltaU, C, beta):
    numerator = np.mean(Fermi(-1.0 * beta * (backward_deltaU - C)))
    denominator = np.mean(Fermi(1.0 * beta * (forward_deltaU - C)))
    deltaA = np.log(numerator / denominator) / beta + C
    return deltaA

def estimateC(deltaA, N_forward, N_backward, beta):
    C = deltaA + 1.0 / beta * np.log(N_backward / N_forward)
    return C

def BAR(forward_deltaU, backward_deltaU, beta, maxIterations, tolerance):
    C = 0
    iteration = 1
    while iteration < maxIterations:
        C_previous = C
        deltaA = estimateDeltaA(forward_deltaU, -1.0 * backward_deltaU, C, beta)
        C = estimateC(deltaA, len(forward_deltaU), len(backward_deltaU), beta)
        error = np.abs(C_previous - C)
        if error < tolerance:
            break
        iteration = iteration + 1
    if iteration >= maxIterations:
        print(f'Warning: BAR does not converge in {maxIterations} iterations!')
    return deltaA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', type=str, help='forward fepout file')
    parser.add_argument('--backward', type=str, help='backward fepout file')
    parser.add_argument('--kbt', default=(300.0*0.0019872041), type=float, help='inverse temperature')
    args = parser.parse_args()

    current_beta = 1.0 / args.kbt
    forward_ensembles = ensemblesFromOutput(args.forward)
    backward_ensembles = ensemblesFromOutput(args.backward)
    showEnsembles(forward_ensembles, 'Forward:')
    showEnsembles(backward_ensembles, 'Backward:')

    deltaA = []
    for i, (forward_deltaU, backward_deltaU) in enumerate(zip(forward_ensembles, reversed(backward_ensembles))):
        deltaA_i = BAR(forward_deltaU, backward_deltaU, current_beta, 10000, 1e-5)
        deltaA.append(deltaA_i)

    print('BAR results:')
    A = 0
    print(A)
    for x in deltaA:
        A += x
        print(A)
    print('All deltaA:')
    print(deltaA)
