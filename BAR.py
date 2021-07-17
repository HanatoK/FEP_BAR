#!/usr/bin/env python3
import numpy as np
import argparse

def ensemblesFromOutput(outputfile):
    ensembles = []
    lambda_intervals = []
    with open(outputfile, 'r') as fInput:
        ensemble = []
        start_collecting = False
        for line in fInput:
            if line.startswith('#NEW FEP WINDOW'):
                ensemble = []
                lambda_start = float(line.split()[6])
                lambda_end = float(line.split()[8])
                lambda_intervals.append(np.array([lambda_start, lambda_end]))
            if line.startswith('FepEnergy:') and start_collecting:
                fields = line.split()
                ensemble.append(float(fields[6]))
            if line.startswith('#STARTING COLLECTION'):
                start_collecting = True
            if line.startswith('#Free energy change'):
                start_collecting = False
                if ensemble:
                    ensembles.append(np.array(ensemble))
    return lambda_intervals, ensembles

# debug the data reading
def showEnsembles(ensembles, title):
    print(title)
    print('#' + '=' * 80)
    print(f'#Number of windows: {len(ensembles)}')
    for i, ensemble in enumerate(ensembles):
        print(f'#Samples in window {i:3d}: {len(ensemble)}')
    print('#' + '=' * 80)
    # print('\n')


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
    parser.add_argument('--kbt', default=(300.0*0.0019872041), type=float, help='temperature multiplied with the Boltzmann constant')
    args = parser.parse_args()

    current_beta = 1.0 / args.kbt
    forward_lambda, forward_ensembles = ensemblesFromOutput(args.forward)
    backward_lambda, backward_ensembles = ensemblesFromOutput(args.backward)
    # print(forward_lambda)
    # print(backward_lambda)
    print('#' + '=' * 80)
    showEnsembles(forward_ensembles, '#Forward:')
    showEnsembles(backward_ensembles, '#Backward:')

    deltaA = []
    for i, (forward_deltaU, backward_deltaU) in enumerate(zip(forward_ensembles, reversed(backward_ensembles))):
        deltaA_i = BAR(forward_deltaU, backward_deltaU, current_beta, 10000, 1e-7)
        deltaA.append(deltaA_i)

    print('#' + ' ' * 18 + 'Bennett Acceptance Ratio (BAR) Estimator')
    print('#' + '=' * 80)
    print(f'#{"λ":>14s} {"Δλ":>18s} {"ΔA":>18s} {"ΔΔA":>18s}')
    lambda_start = forward_lambda[0][0]
    print(f'{lambda_start:>15.7f} {" ":>18s} {0:>18.7f} {" ":>18s}')
    A = 0
    for i,x in enumerate(deltaA):
        lambda_start = forward_lambda[i][1]
        delta_lambda = forward_lambda[i][1] - forward_lambda[i][0]
        A += x
        print(f'{lambda_start:>15.7f} {delta_lambda:>18.7f} {A:>18.7f} {x:>18.7f}')
    print('#' + '=' * 80)
