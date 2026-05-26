#include <stdint.h>
#include <stdbool.h>

/**
 * Stateful Peak-bell IIR filter.
 * Use `FilterIIRPeakBell<f64>` for audio, otherwise you can use `FilterIIRPeakBell<Complex64>`
 */
typedef struct FilterIIRPeakBell_f64 FilterIIRPeakBell_f64;

typedef struct Tuple3F64 {
  double a;
  double b;
  double c;
} Tuple3F64;

struct FilterIIRPeakBell_f64 *new_FilterIIRPeakBellF64(const struct Tuple3F64 *bands_ptr,
                                                       uintptr_t bands_count,
                                                       uintptr_t sample_rate);

void free_FilterIIRPeakBellF64(struct FilterIIRPeakBell_f64 *ptr);

double process_sample_FilterIIRPeakBellF64(struct FilterIIRPeakBell_f64 *filter, double sample);
