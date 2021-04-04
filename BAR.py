#!/usr/bin/env python3
import numpy as np

temperature = 300.0
boltzmann_constant = 0.0019872041
current_beta = 1.0 / (temperature * boltzmann_constant)

def ensemblesFromOutput(outputfile):
    ensembles = []
    with open(outputfile, 'r') as fInput:
        ensemble = []
        start_collecting = False
        for line in fInput:
            if line.startswith('#NEW FEP WINDOW'):
                if ensemble:
                    ensembles.append(np.array(ensemble))
                ensemble = []
            if line.startswith('FepEnergy:') and start_collecting:
                fields = line.split()
                ensemble.append(float(fields[6]))
            if line.startswith('#STARTING COLLECTION'):
                start_collecting = True
            if line.startswith('#Free energy change'):
                start_collecting = False
    return ensembles

forward_ensembles = ensemblesFromOutput('forward-shift-long.fepout')
backward_ensembles = ensemblesFromOutput('backward-shift-long.fepout')

# debug the data reading
def showEnsembles(ensembles, title):
    print(title)
    print('=' * 80)
    print(f'Number of windows: {len(ensembles)}')
    for i, ensemble in enumerate(ensembles):
        print(f'Samples in window {i:3d}: {len(ensemble)}')
        #print(ensemble)
    print('=' * 80)
    print('\n')

showEnsembles(forward_ensembles, 'Forward:')
showEnsembles(backward_ensembles, 'Backward:')

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
        #print(f'Tteration {iteration}: error = {error:15.7f} ; C = {C:15.7f} ; deltaA = {deltaA:15.7f}')
        if error < tolerance:
            break
        iteration = iteration + 1
    if iteration >= maxIterations:
        print(f'Warning: BAR does not converge in {maxIterations} iterations!')
    return deltaA

deltaA = []
for i, (forward_deltaU, backward_deltaU) in enumerate(zip(forward_ensembles, reversed(backward_ensembles))):
    deltaA_i = BAR(forward_deltaU, backward_deltaU, current_beta, 10000, 1e-9)
    deltaA.append(deltaA_i)

print('BAR results:')
A = 0
print(A)
for x in deltaA:
    A += x
    print(A)
print('ALl deltaA:')
print(deltaA)
