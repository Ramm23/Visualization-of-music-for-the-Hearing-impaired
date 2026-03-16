#ifndef YIN_H
#define YIN_H

#include <stdbool.h>

#ifdef _WIN32
#define YIN_EXPORT __declspec(dllexport)
#else
#define YIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        float time_s;
        float frequency_hz;
        float confidence;
        char note[5];
        int octave;
        char note_with_octave[16];
        int midi;
        float cents_error;
        float rms;
        bool in_key;
    } FrameAnnotation;

    YIN_EXPORT void process_audio_frames(
        const float *audio,
        int num_samples,
        int sr,
        int tonic_pc,
        const int *scale_intervals,
        int num_scale_intervals,
        float fmin,
        float fmax,
        int frame_size,
        int hop_size,
        float thresh,
        float conf_thresh,
        FrameAnnotation **out_annotations,
        int *out_count);

    YIN_EXPORT void free_annotations(FrameAnnotation *annotations);

#ifdef __cplusplus
}
#endif

#endif