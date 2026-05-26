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

/**
 * Instantiate a `FilterIIRPeakBell<f64>` with bands' frequency defined by Tuple3F64 a, gain Tuple3F64 b, and Q factor Tuple3F64 c.
 */
struct FilterIIRPeakBell_f64 *new_FilterIIRPeakBellF64(const struct Tuple3F64 *bands_ptr,
                                                       uintptr_t bands_count,
                                                       uintptr_t sample_rate);

void free_FilterIIRPeakBellF64(struct FilterIIRPeakBell_f64 *ptr);

double process_sample_FilterIIRPeakBellF64(struct FilterIIRPeakBell_f64 *filter, double sample);
