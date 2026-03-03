#ifndef YIN_H
#define YIN_H

#ifdef _WIN32
#define YIN_EXPORT __declspec(dllexport)
#else
#define YIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct YinContext
    {
        int max_tau;   // maximum tau supported (buffer size - 1)
        float *d;      // difference function buffer [0..max_tau]
        float *dprime; // CMNDF buffer [0..max_tau]
    } YinContext;

    typedef struct YinPitch
    {
        float f0;           // estimated fundamental frequency (Hz)
        float tau;          // estimated period (samples), fractional after interpolation
        float confidence;   // simple proxy: 1 - aperiodicity
        float aperiodicity; // d'(tau_int) at chosen tau (lower is better)
        int t_used;         // frame start index actually used (after step 6)
    } YinPitch;

    // Allocate reusable buffers for up to max_tau (inclusive)
    YIN_EXPORT YinContext *yin_create(int max_tau);

    // Free buffers
    YIN_EXPORT void yin_destroy(YinContext *ctx);

    /**
     * Detect pitch using YIN (steps 2–6).
     *
     * signal: pointer to mono float samples
     * length: number of samples in signal
     * sr: sample rate (Hz)
     * t: frame start sample index
     * W: window length (samples)
     * tau_min/tau_max: lag search bounds (samples)
     * thresh: absolute threshold (typical ~0.1)
     *
     * Step 6 params:
     * local_radius: +/- samples around t to search (0 disables step 6)
     * local_step: stride within neighborhood
     * refine_radius: second pass tight search around best tau (in lag samples)
     *
     * out: result struct
     *
     * returns 1 on success, 0 if not enough samples / invalid params
     */
    YIN_EXPORT int yin_detect_pitch(
        YinContext *ctx,
        const float *signal,
        int length,
        float sr,
        int t,
        int W,
        int tau_min,
        int tau_max,
        float thresh,
        int local_radius,
        int local_step,
        int refine_radius,
        YinPitch *out);

#ifdef __cplusplus
}
#endif

#endif // YIN_H