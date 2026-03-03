#ifndef VOLUME_H
#define VOLUME_H

#ifdef _WIN32
#define VOL_EXPORT __declspec(dllexport)
#else
#define VOL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct VolumeResult
    {
        float rms;    // linear RMS
        float db;     // dB (typically dBFS if samples are [-1,1])
        float norm;   // normalized 0..1 after floor/ceiling clamp
        int discrete; // mapped integer level
    } VolumeResult;

    /**
     * Computes RMS->dB->normalize->discretize for a buffer.
     *
     * signal: float samples (mono). If stereo, pass one channel or pre-mix.
     * length: number of samples
     * db_floor: e.g. -60.0f
     * db_ceil:  e.g. 0.0f
     * bins: number of discrete levels (e.g. 128, 101, 11). Must be >= 2.
     *
     * Returns 1 on success, 0 on invalid inputs.
     */
    VOL_EXPORT int volume_analyze(
        const float *signal,
        int length,
        float db_floor,
        float db_ceil,
        int bins,
        VolumeResult *out);

#ifdef __cplusplus
}
#endif

#endif // VOLUME_H