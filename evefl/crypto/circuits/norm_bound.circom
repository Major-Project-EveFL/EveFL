// NormBound circuit (n=8 test size)
// Private input: g[8]  -- fixed-point-scaled integer gradient components (hidden)
// Public output: S      -- sum of squares of g (revealed; caller checks S <= tau^2 outside the circuit)
//
// This proves "I know a vector g such that sum(g_i^2) = S" without
// revealing any individual g_i. Scoped this way (revealing only the
// scalar norm, not a full range-proof in-circuit) to avoid needing a
// comparator component, while still being a genuine ZK statement:
// the gradient values themselves never leave the client.

template NormBound(n) {
    signal private input g[n];
    signal output S;

    signal sq[n];
    signal partial[n];

    for (var i = 0; i < n; i++) {
        sq[i] <== g[i] * g[i];
    }

    partial[0] <== sq[0];
    for (var i = 1; i < n; i++) {
        partial[i] <== partial[i-1] + sq[i];
    }

    S <== partial[n-1];
}

component main = NormBound(8);