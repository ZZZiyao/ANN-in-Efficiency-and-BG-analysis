import sys
import tensorflow as tf
import amplitf.interface as atfi
import tfa.rootio as tfr

from DistributionModel import (
    observables_phase_space,
    selection_mask,
    true_cuts,
    random_array_size
)

def main():

    nev = int(sys.argv[1])      # number of events
    outfile = sys.argv[2]

    chunk_size = 500000
    atfi.set_seed(nev)

    arrays = []
    n_tot = 0

    while n_tot < nev:

        # 1️⃣ generate flat (m', theta') in square Dalitz
        unfiltered = observables_phase_space.unfiltered_sample(chunk_size)
        sample = observables_phase_space.filter(unfiltered)

        size = sample.shape[0]

        # 2️⃣ random numbers for kinematics
        rnd = tf.random.uniform(
            [size, random_array_size],
            dtype=atfi.fptype()
        )

        # 3️⃣ compute selection mask (NO filtering)
        sel = selection_mask(sample, true_cuts, rnd)
        pass_flag = tf.cast(sel, atfi.fptype())

        # 4️⃣ stack output
        mprime = sample[:, 0]
        thetaprime = sample[:, 1]

        out = atfi.stack([mprime, thetaprime, pass_flag], axis=1)

        arrays.append(out)
        n_tot += size

        print(f"Generated so far: {n_tot}")

    # concatenate and truncate
    final_array = atfi.concat(arrays, axis=0)[:nev, :]

    # write to ROOT
    observables = ["mprime", "thetaprime", "pass"]

    tfr.write_tuple(outfile, final_array, observables)

    print("Done. File written:", outfile)


if __name__ == "__main__":
    main()