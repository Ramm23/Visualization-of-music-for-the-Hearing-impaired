using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class YinPitchTracker : MonoBehaviour
{
    // --- 1. Define the C# Struct to match the C Struct ---
    // CharSet.Ansi ensures our char[] strings map correctly
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct FrameAnnotation
    {
        public float time_s;
        public float frequency_hz;
        public float confidence;
        
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 5)] 
        public string note;
        
        public int octave;
        
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 10)] 
        public string note_with_octave;
        
        public int midi;
        public float cents_error;
        
        [MarshalAs(UnmanagedType.I1)] // Forces 1-byte boolean to match C's stdbool.h
        public bool in_key;
    }

    // --- 2. DLL Imports ---
    // Change "YinPlugin" to the exact name of your compiled DLL/SO/Bundle (without the extension)
    private const string pluginName = "YinPlugin";

    [DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void process_audio_frames(
        float[] audio, int num_samples, int sr,
        int tonic_pc, int[] scale_intervals, int num_scale_intervals,
        float fmin, float fmax, int frame_size, int hop_size, float thresh, float conf_thresh,
        out IntPtr out_annotations, out int out_count
    );

    [DllImport(pluginName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void free_annotations(IntPtr annotations);

    // --- 3. Unity Integration ---
    
    [Header("Audio Settings")]
    public AudioClip audioClip;
    
    [Header("YIN Settings")]
    public float fMin = 40f;
    public float fMax = 2000f;
    public int frameSize = 2048;
    public int hopSize = 1024;
    public float threshold = 0.1f;
    public float confidenceThreshold = 0.6f;

    // Standard C Major scale intervals
    private readonly int[] majorScale = { 0, 2, 4, 5, 7, 9, 11 };

    void Start()
    {
        if (audioClip != null)
        {
            AnalyzeClip(audioClip);
        }
        else
        {
            Debug.LogWarning("Please assign an AudioClip to analyze.");
        }
    }

    public void AnalyzeClip(AudioClip clip)
    {
        // 1. Extract raw float data from the Unity AudioClip
        float[] audioData = new float[clip.samples * clip.channels];
        clip.GetData(audioData, 0);

        // If the clip is stereo, you'll ideally want to mix it down to mono first. 
        // For simplicity, this assumes a Mono clip or directly processes the interleaved data.
        
        IntPtr annotationsPtr = IntPtr.Zero;
        int frameCount = 0;

        // 2. Call the C function
        process_audio_frames(
            audioData, audioData.Length, clip.frequency,
            0, majorScale, majorScale.Length, // C Major config
            fMin, fMax, frameSize, hopSize, threshold, confidenceThreshold,
            out annotationsPtr, out frameCount
        );

        if (frameCount > 0 && annotationsPtr != IntPtr.Zero)
        {
            // 3. Marshal the unmanaged memory into a managed C# array
            FrameAnnotation[] annotations = new FrameAnnotation[frameCount];
            int structSize = Marshal.SizeOf(typeof(FrameAnnotation));

            for (int i = 0; i < frameCount; i++)
            {
                // Calculate memory offset for each struct in the array
                IntPtr currentPtr = new IntPtr(annotationsPtr.ToInt64() + (i * structSize));
                annotations[i] = Marshal.PtrToStructure<FrameAnnotation>(currentPtr);
            }

            // 4. MUST FREE C MEMORY! Prevent memory leaks
            free_annotations(annotationsPtr);

            // 5. Test the output
            Debug.Log($"Successfully processed {frameCount} frames.");
            for (int i = 0; i < Mathf.Min(frameCount, 20); i++)
            {
                if (annotations[i].midi != -1) // Only log voiced/confident frames
                {
                    Debug.Log($"Time: {annotations[i].time_s:F2}s | Note: {annotations[i].note_with_octave} | Freq: {annotations[i].frequency_hz:F1}Hz");
                }
            }
        }
    }
}