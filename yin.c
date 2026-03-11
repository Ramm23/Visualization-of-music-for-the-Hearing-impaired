#include "yin.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

// Constants and Data
static const char *NOTE_NAMES_SHARP[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
static const int SCALE_MAJOR[] = {0, 2, 4, 5, 7, 9, 11};

// --- Internal Helper Functions ---

static float hz_to_midi(float freq)
{
    if (freq <= 0)
        return 0;
    return 69.0f + 12.0f * log2f(freq / 440.0f);
}

static uint16_t build_key_mask(int tonic_pc, const int *intervals, int num_intervals)
{
    uint16_t mask = 0;
    for (int i = 0; i < num_intervals; i++)
    {
        mask |= (1 << ((tonic_pc + intervals[i]) % 12));
    }
    return mask;
}

// --- YIN Core Steps ---

void difference_function(const float *buffer, int W, int tau_max, float *d)
{
    for (int tau = 0; tau <= tau_max; tau++)
    {
        d[tau] = 0.0f;
        for (int j = 0; j < W; j++)
        {
            float tmp = buffer[j] - buffer[j + tau];
            d[tau] += tmp * tmp;
        }
    }
}

void cmndf(float *d, int tau_max)
{
    d[0] = 1.0f;
    float running_sum = 0.0f;
    for (int tau = 1; tau <= tau_max; tau++)
    {
        running_sum += d[tau];
        if (running_sum > 0)
            d[tau] = d[tau] / (running_sum / tau);
        else
            d[tau] = 1.0f;
    }
}

int absolute_threshold(float *d, int tau_min, int tau_max, float thresh)
{
    for (int tau = tau_min; tau <= tau_max; tau++)
    {
        if (d[tau] < thresh)
        {
            while (tau + 1 <= tau_max && d[tau + 1] < d[tau])
                tau++;
            return tau;
        }
    }
    int min_tau = tau_min;
    for (int tau = tau_min + 1; tau <= tau_max; tau++)
    {
        if (d[tau] < d[min_tau])
            min_tau = tau;
    }
    return min_tau;
}

float parabolic_interpolation(const float *d, int tau, int tau_max)
{
    if (tau <= 0 || tau >= tau_max)
        return (float)tau;
    float s0 = d[tau - 1], s1 = d[tau], s2 = d[tau + 1];
    float denom = s0 - 2.0f * s1 + s2;
    if (fabsf(denom) < 1e-20f)
        return (float)tau;
    return (float)tau + 0.5f * (s0 - s2) / denom;
}

// --- Primary Entry Point (Unity & Main) ---

void process_audio_frames(
    const float *audio, int num_samples, int sr,
    int tonic_pc, const int *scale_intervals, int num_scale_intervals,
    float fmin, float fmax, int frame_size, int hop_size, float thresh, float conf_thresh,
    FrameAnnotation **out_annotations, int *out_count)
{
    int tau_min = (int)fmaxf(2.0f, (float)sr / fmax);
    int tau_max = (int)((float)sr / fmin);

    int max_start = num_samples - (frame_size + tau_max);
    if (max_start <= 0)
    {
        *out_count = 0;
        return;
    }

    int total_frames = (max_start / hop_size) + 1;
    *out_annotations = (FrameAnnotation *)malloc(total_frames * sizeof(FrameAnnotation));
    *out_count = total_frames;

    float *d = (float *)malloc((tau_max + 1) * sizeof(float));
    uint16_t key_mask = build_key_mask(tonic_pc, scale_intervals, num_scale_intervals);

    for (int i = 0; i < total_frames; i++)
    {
        int start = i * hop_size;
        difference_function(&audio[start], frame_size, tau_max, d);
        cmndf(d, tau_max);

        int tau_int = absolute_threshold(d, tau_min, tau_max, thresh);
        float tau_hat = parabolic_interpolation(d, tau_int, tau_max);

        float aperiodicity = d[tau_int];
        float confidence = fmaxf(0.0f, fminf(1.0f, 1.0f - aperiodicity));
        float f0 = (tau_hat > 0.0f) ? ((float)sr / tau_hat) : 0.0f;

        FrameAnnotation *ann = &(*out_annotations)[i];
        ann->time_s = (float)start / sr;
        ann->confidence = confidence;

        if (confidence >= conf_thresh && f0 >= fmin && f0 <= fmax)
        {
            ann->frequency_hz = f0;
            float midi_float = hz_to_midi(f0);
            int midi_round = (int)roundf(midi_float);

            ann->cents_error = (midi_float - (float)midi_round) * 100.0f;
            ann->midi = midi_round;
            int pitch_class = midi_round % 12;
            ann->octave = (midi_round / 12) - 1;

            strncpy(ann->note, NOTE_NAMES_SHARP[pitch_class], 5);
            snprintf(ann->note_with_octave, 10, "%s%d", ann->note, ann->octave);
            ann->in_key = (key_mask & (1 << pitch_class)) != 0;
        }
        else
        {
            ann->frequency_hz = -1.0f;
            ann->midi = -1;
            ann->octave = -1;
            strcpy(ann->note, "-");
            strcpy(ann->note_with_octave, "-");
            ann->in_key = false;
        }
    }
    free(d);
}

void free_annotations(FrameAnnotation *annotations)
{
    if (annotations)
        free(annotations);
}

// --- Improved WAV Loader (Correctly finds DATA chunk) ---

float *load_wav_mono(const char *filename, int *out_num_samples, int *out_sr)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return NULL;

    char chunk_id[4];
    uint32_t chunk_size;

    // Read RIFF Header
    fseek(f, 12, SEEK_SET);

    // Search for "fmt " chunk
    while (fread(chunk_id, 1, 4, f) == 4)
    {
        fread(&chunk_size, 4, 1, f);
        if (memcmp(chunk_id, "fmt ", 4) == 0)
        {
            fseek(f, 4, SEEK_CUR); // skip audio format
            uint16_t channels;
            fread(&channels, 2, 1, f);
            uint32_t sr;
            fread(&sr, 4, 1, f);
            *out_sr = (int)sr;
            fseek(f, chunk_size - 10, SEEK_CUR); // skip rest of fmt
        }
        else if (memcmp(chunk_id, "data", 4) == 0)
        {
            int16_t *pcm_data = (int16_t *)malloc(chunk_size);
            int read = fread(pcm_data, 1, chunk_size, f);
            int samples = read / 2;
            float *float_data = (float *)malloc(samples * sizeof(float));
            for (int i = 0; i < samples; i++)
                float_data[i] = pcm_data[i] / 32768.0f;
            *out_num_samples = samples;
            free(pcm_data);
            fclose(f);
            return float_data;
        }
        else
        {
            fseek(f, chunk_size, SEEK_CUR); // Skip unknown chunks
        }
    }
    fclose(f);
    return NULL;
}

int main()
{
    const char *filepath = "Happy-birthday-piano-music.wav";
    int num_samples = 0, sr = 0;

    float *audio = load_wav_mono(filepath, &num_samples, &sr);
    if (!audio)
    {
        printf("Error: Could not load WAV file.\n");
        return 1;
    }

    printf("Loaded audio: %d samples at %d Hz\n", num_samples, sr);

    FrameAnnotation *annotations = NULL;
    int num_frames = 0;

    process_audio_frames(
        audio, num_samples, sr,
        0, SCALE_MAJOR, 7,
        40.0f, 2000.0f,
        2048, 1024,
        0.1f, 0.6f,
        &annotations, &num_frames);

    printf("\nDetected Frames (First 20 voiced):\n");
    int printed = 0;
    for (int i = 0; i < num_frames && printed < 20; i++)
    {
        if (annotations[i].midi != -1)
        {
            printf("Time: %5.2fs | Freq: %7.1f Hz | Note: %4s | In-Key: %d | Conf: %.2f\n",
                   annotations[i].time_s, annotations[i].frequency_hz,
                   annotations[i].note_with_octave, annotations[i].in_key, annotations[i].confidence);
            printed++;
        }
    }

    free_annotations(annotations);
    free(audio);
    return 0;
}