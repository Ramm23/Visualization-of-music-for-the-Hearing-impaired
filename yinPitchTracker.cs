using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class YinPitchTracker : MonoBehaviour
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct FrameAnnotation
    {
        public float time_s;
        public float frequency_hz;
        public float confidence;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 5)]
        public string note;

        public int octave;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 16)]
        public string note_with_octave;

        public int midi;
        public float cents_error;
        public float rms;

        [MarshalAs(UnmanagedType.I1)]
        public bool in_key;
    }

    [DllImport("yin", CallingConvention = CallingConvention.Cdecl)]
    private static extern void process_audio_frames(
        float[] audio, int num_samples, int sr,
        int tonic_pc, int[] scale_intervals, int num_scale_intervals,
        float fmin, float fmax, int frame_size, int hop_size, float thresh, float conf_thresh,
        out IntPtr out_annotations, out int out_count);

    [DllImport("yin", CallingConvention = CallingConvention.Cdecl)]
    private static extern void free_annotations(IntPtr annotations);

    public FrameAnnotation[] AnalyzeClip(
    AudioClip clip,
    int tonicPc,
    int[] scaleIntervals,
    float fmin,
    float fmax,
    int frameSize,
    int hopSize,
    float thresh,
    float confThresh)
{
    if (clip == null)
        throw new ArgumentNullException(nameof(clip));

    float[] interleaved = new float[clip.samples * clip.channels];
    clip.GetData(interleaved, 0);

    float[] mono = new float[clip.samples];

    if (clip.channels == 1)
    {
        Array.Copy(interleaved, mono, clip.samples);
    }
    else
    {
        for (int i = 0; i < clip.samples; i++)
        {
            float sum = 0f;
            for (int ch = 0; ch < clip.channels; ch++)
            {
                sum += interleaved[i * clip.channels + ch];
            }
            mono[i] = sum / clip.channels;
        }
    }

    IntPtr annotationsPtr;
    int count;

    process_audio_frames(
        mono,
        mono.Length,
        clip.frequency,
        tonicPc,
        scaleIntervals,
        scaleIntervals.Length,
        fmin,
        fmax,
        frameSize,
        hopSize,
        thresh,
        confThresh,
        out annotationsPtr,
        out count);

    if (count <= 0 || annotationsPtr == IntPtr.Zero)
        return Array.Empty<FrameAnnotation>();

    var result = new FrameAnnotation[count];
    int structSize = Marshal.SizeOf<FrameAnnotation>();

    for (int i = 0; i < count; i++)
    {
        IntPtr curr = IntPtr.Add(annotationsPtr, i * structSize);
        result[i] = Marshal.PtrToStructure<FrameAnnotation>(curr);
    }

    free_annotations(annotationsPtr);
    return result;
}
}