states = ['Rainy', 'Sunny']
observations = ['walk', 'shop', 'clean']

transition_prob = {
    'Rainy': {
        'Rainy': 0.7,
        'Sunny': 0.3
    },
    'Sunny': {
        'Rainy': 0.4,
        'Sunny': 0.6
    }
}

emission_prob = {
    'Rainy': {
        'walk': 0.1,
        'shop': 0.4,
        'clean': 0.5
    },
    'Sunny': {
        'walk': 0.6,
        'shop': 0.3,
        'clean': 0.1
    }
}

start_prob = {
    'Rainy': 0.6,
    'Sunny': 0.4
}

obs_seq = ['walk', 'shop', 'clean']


def forward(obs_seq):
    fwd = [{}]

    for s in states:
        fwd[0][s] = start_prob[s] * emission_prob[s][obs_seq[0]]

    for t in range(1, len(obs_seq)):
        fwd.append({})

        for s in states:
            fwd[t][s] = (
                sum(
                    fwd[t - 1][ps] * transition_prob[ps][s]
                    for ps in states
                )
                * emission_prob[s][obs_seq[t]]
            )

    return sum(fwd[-1][s] for s in states)


def viterbi(obs_seq):
    V = [{}]
    path = {}

    for s in states:
        V[0][s] = start_prob[s] * emission_prob[s][obs_seq[0]]
        path[s] = [s]

    for t in range(1, len(obs_seq)):
        V.append({})
        newpath = {}

        for s in states:
            prob, state = max(
                (
                    V[t - 1][ps]
                    * transition_prob[ps][s]
                    * emission_prob[s][obs_seq[t]],
                    ps
                )
                for ps in states
            )

            V[t][s] = prob
            newpath[s] = path[state] + [s]

        path = newpath

    n = len(obs_seq) - 1

    prob, state = max(
        (V[n][s], s)
        for s in states
    )

    return prob, path[state]


print("Forward Probability of sequence:", forward(obs_seq))

v_prob, v_path = viterbi(obs_seq)

print("Viterbi best path probability:", v_prob)
print("Most likely hidden states:", v_path)
