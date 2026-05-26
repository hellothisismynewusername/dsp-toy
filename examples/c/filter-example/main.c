#include "include/dsp_toy_lib.h"
#include <stdio.h>

int main() {
    printf("Example start\n");
    double signal[] = {0.279, -0.949, 0.001, -0.595, -0.100, 1.000, -0.140, 0.709};
    int signal_len = 8;

    Tuple3F64 band = {
        2.0,
        -50.0,
        1.0
    };

    FilterIIRPeakBell_f64 *filter = new_FilterIIRPeakBellF64(&band, 1, 8);

    for (int i = 0; i < signal_len; i++) {
        printf("Filtered sample %d: %f\n", i, process_sample_FilterIIRPeakBellF64(filter, signal[i]));
    }

    free_FilterIIRPeakBellF64(filter);
    printf("Example end");
}